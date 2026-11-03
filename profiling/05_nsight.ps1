# Step 5 - Nsight before/after: timelines (nsys) and per-kernel deep dives (ncu) for stock vs custom kernels.
#
# Prereqs (README section 4): Nsight Systems + Nsight Compute installed and on PATH, and GPU performance
# counters allowed for all users (NVIDIA Control Panel > Developer > Manage GPU Performance Counters).
#
# Run from the project root:   powershell -ExecutionPolicy Bypass -File profiling/05_nsight.ps1 [-Batch 4]
param([int]$Batch = 4, [int]$Steps = 16)

$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
$target = "profiling\03_nsight_target.py"
$env:SDPA_BACKEND = "cudnn"
New-Item -ItemType Directory -Force results | Out-Null

foreach ($variant in @("stock", "opt")) {
    $env:OPTIMIZED = if ($variant -eq "opt") { "1" } else { "0" }
    $tag = "bs${Batch}_$variant"

    # 1) Timeline: CPU launches, kernels, gaps. Capture starts at cudaProfilerStart (after load/warm-up/calibration).
    nsys profile -t cuda,nvtx,cublas --capture-range=cudaProfilerApi --capture-range-end=stop `
        --cuda-memory-usage=true -o "results\nsys_$tag" -f true $py $target --batch $Batch --steps $Steps
    nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv `
        --output "results\nsys_$tag" "results\nsys_$tag.nsys-rep"

    # 2) Kernel deep dive: matmul + attention kernels inside the 2nd decode step only.
    #    stock: bitsandbytes gemm_4bit*, SDPA math kernels;  opt: _nf4_gemm_kernel, _partial_kernel, _merge_kernel
    $regex = if ($variant -eq "opt") { "regex:_nf4_gemm|_partial_kernel|_merge_kernel|gemm_4bit" } else { "regex:gemm_4bit|gemv|softmax|elementwise" }
    ncu --nvtx --nvtx-include "decode_step/" -k $regex --launch-skip 0 --launch-count 30 `
        --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats `
        -o "results\ncu_$tag" -f $py $target --batch $Batch --steps 2
}
Write-Host "Done. Open results\nsys_*.nsys-rep in Nsight Systems and results\ncu_*.ncu-rep in Nsight Compute."
