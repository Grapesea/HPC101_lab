from _bootstrap import add_local_src_to_path

add_local_src_to_path()

from hpc101_infer.quantization.cli import main


if __name__ == "__main__":
    main()
