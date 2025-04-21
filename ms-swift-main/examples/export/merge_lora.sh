# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
swift export \
    --adapters "D:\UBC\eece570\output\v12-20250419-004804\checkpoint-51376" \
    --merge_lora true
