from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from itertools import count
from queue import Empty, Queue
from threading import Lock
from typing import Callable, Iterable

from hpc101_infer.engine import InferenceEngine
from hpc101_infer.types import GenerationOutput, GenerationRequest


@dataclass(frozen=True)
class RequestHandle:
    request_id: int
    _future: Future[GenerationOutput] = field(repr=False, compare=False)

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> GenerationOutput:
        return self._future.result(timeout=timeout)


@dataclass(frozen=True)
class CompletedRequest:
    request_id: int
    output: GenerationOutput


@dataclass(frozen=True)
class PendingRequest:
    handle: RequestHandle
    request: GenerationRequest


class Runner:
    def __init__(self, engine: InferenceEngine) -> None:
        self.engine = engine
        self._pending: Queue[PendingRequest] = Queue()
        self._request_ids = count()
        self._submit_lock = Lock()
        self._step_lock = Lock()

    @property
    def pending_count(self) -> int:
        return self._pending.qsize()

    def submit(self, request: GenerationRequest) -> RequestHandle:
        with self._submit_lock:
            request_id = next(self._request_ids)
        handle = RequestHandle(request_id, Future())
        self._pending.put_nowait(PendingRequest(handle, request))
        return handle

    def has_pending_requests(self) -> bool:
        return not self._pending.empty()

    def step(self) -> list[CompletedRequest]:
        with self._step_lock:
            try:
                batch = [self._pending.get_nowait()]
            except Empty:
                return []

            # Fill the batch with pending requests up to the scheduler batch size
            for _ in range(self.engine.config.scheduler_batch_size - 1):
                try:
                    batch.append(self._pending.get_nowait())
                except Empty:
                    break

            try:
                # Run the inference engine on the batch of requests
                outputs = self.engine.generate([item.request for item in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError("engine output count does not match batch size")
                completed = []
                for item, output in zip(batch, outputs, strict=True):
                    item.handle._future.set_result(output)
                    completed.append(CompletedRequest(item.handle.request_id, output))
                return completed
            except BaseException as error:
                for item in batch:
                    item.handle._future.set_exception(error)
                raise
            finally:
                for _ in batch:
                    self._pending.task_done()

    def drain(
        self,
        *,
        on_completed: Callable[[int], None] | None = None,
    ) -> list[CompletedRequest]:
        completed = []
        while self.has_pending_requests():
            completed_batch = self.step()
            completed.extend(completed_batch)
            if completed_batch and on_completed is not None:
                on_completed(len(completed_batch))
        return completed

    def run(
        self,
        requests: Iterable[GenerationRequest],
        *,
        on_completed: Callable[[int], None] | None = None,
    ) -> list[GenerationOutput]:
        # 评测采用无限并发：所有请求在推理开始前都已到达。将长度相近的
        # 生成长度相近的请求放进同一静态 batch，避免短请求等待同批长请求
        # 时仍执行无效 decode；再以 prompt 长度作为次级排序以减少 padding。
        # handle 数组仍按原始输入顺序返回。
        request_list = list(requests)

        def prompt_size(index: int) -> int:
            request = request_list[index]
            if request.input_ids is not None:
                return len(request.input_ids)
            return len(request.prompt or "")

        ordered_indices = sorted(
            range(len(request_list)),
            key=lambda index: (
                prompt_size(index),
                request_list[index].max_new_tokens,
            ),
            reverse=True,
        )
        handles: list[RequestHandle | None] = [None] * len(request_list)
        for index in ordered_indices:
            handles[index] = self.submit(request_list[index])
        self.drain(on_completed=on_completed)
        return [handle.result() for handle in handles if handle is not None]
