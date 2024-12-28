# run_tests.py

import os
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt

# Paths
target_binary = "./bin/cuda_convolution"
input_dir = "./input"
output_dir = "./output"
graphs_dir = "./graphs"
tables_dir = "./tables"

# Create directories if they don't exist
os.makedirs(graphs_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Test parameters
images = [
    ("1200x1200.png", 1200, 1200),
    ("720x720.png", 720, 720),
    ("360x360.png", 360, 360)
]
block_sizes = [16, 32, 64]
grid_shapes = [
    (10, 10),  # Square
    (20, 10),  # Rectangle
    (15, 30),  # Vertical Rectangle
    (40, 40)   # Large Square
]


# Compile CUDA code using Makefile
def compile_cuda():
    compile_command = ["make", "all"]
    subprocess.run(compile_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# Run CUDA convolution and record execution times
def run_test(input_image, mode, block_size, grid_shape):
    output_file = f"{output_dir}/{os.path.splitext(input_image)[0]}_{mode}_{block_size}_{grid_shape[0]}x{grid_shape[1]}.png"
    
    command = [
        target_binary,
        f"{input_dir}/{input_image}",
        output_file,
        str(mode),
        str(block_size),
        str(grid_shape[0]),
        str(grid_shape[1])
    ]
    start_time = time.time()
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end_time = time.time()

    return end_time - start_time


# Collect performance data
def main():
    compile_cuda()
    data = []

    for image_name, width, height in images:
        for block_size in block_sizes:
            for grid_shape in grid_shapes:
                for mode in [0, 1]:  # 0: global memory, 1: shared memory
                    exec_time = run_test(image_name, mode, block_size, grid_shape)
                    data.append({
                        "Image": image_name,
                        "Mode": "Global" if mode == 0 else "Shared",
                        "Block Size": block_size,
                        "Grid Shape": f"{grid_shape[0]}x{grid_shape[1]}",
                        "Execution Time": exec_time
                    })

    # Save performance data to CSV
    performance_df = pd.DataFrame(data)
    performance_df.to_csv(f"{tables_dir}/performance_results.csv", index=False)

    # Plot graphs for each block size
    for image_name, _, _ in images:
        df = performance_df[performance_df["Image"] == image_name]

        for block_size in block_sizes:
            plt.figure(figsize=(10, 6))
            markers = {"Global": "o", "Shared": "s"}
            colors = {"Global": "blue", "Shared": "orange"}

            for mode in ["Global", "Shared"]:
                mode_df = df[(df["Mode"] == mode) & (df["Block Size"] == block_size)]
                plt.plot(
                    mode_df["Grid Shape"],
                    mode_df["Execution Time"],
                    marker=markers[mode],
                    label=f"{mode} - Grid Shape {block_size}",
                    color=colors[mode]
                )

            plt.title(f"Performance Comparison for {image_name} - Block Size {block_size}")
            plt.xlabel("Grid Shape")
            plt.ylabel("Execution Time (s)")
            plt.legend(title="Mode", loc="upper left")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{graphs_dir}/{os.path.splitext(image_name)[0]}_block_{block_size}_performance.png")

    # Plot graph for average performance
    avg_df = performance_df.groupby(["Mode", "Grid Shape"])['Execution Time'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    markers = {"Global": "o", "Shared": "s"}
    colors = {"Global": "blue", "Shared": "orange"}

    for mode in ["Global", "Shared"]:
        mode_df = avg_df[avg_df["Mode"] == mode]
        plt.plot(
            mode_df["Grid Shape"],
            mode_df["Execution Time"],
            marker=markers[mode],
            label=f"{mode} (Average)",
            color=colors[mode]
        )

    plt.title("Average Performance Comparison Across All Images")
    plt.xlabel("Grid Shape")
    plt.ylabel("Execution Time (s)")
    plt.legend(title="Mode", loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{graphs_dir}/average_performance.png")

    print("Performance tests completed. Results saved in graphs and tables directories.")


if __name__ == "__main__":
    main()
