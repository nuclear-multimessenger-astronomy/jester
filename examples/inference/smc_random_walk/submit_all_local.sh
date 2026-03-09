#!/bin/bash
# Submit all SMC-RandomWalk inference jobs - Sequential Mode
# Cegah JAX langsung makan 90% VRAM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Optional: kalau masih OOM, paksa clear cache XLA
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
for dir in */; do
    dir=${dir%/}

    if [ -f "$dir/submit.sh" ]; then
        echo "------------------------------------------"
        echo "Processing: $dir (Starting at $(date))"
        echo "------------------------------------------"
        
        # Masuk ke directory pakai subshell biar lebih clean
        (
            cd "$dir" || exit 1
            
            # Jalanin JAX inference secara sinkron
            # Pastikan run_jester_inference tidak berjalan di background (&)
            run_jester_inference config.yaml
            
            # Cek apakah process sebelumnya sukses
            if [ $? -eq 0 ]; then
                echo "Success: $dir finished."
            else
                echo "Error: Inference failed in $dir. Stopping loop to save GPU."
                exit 1
            fi
        ) || exit 1 # Exit script kalau subshell gagal
    fi
done

echo "Done!"