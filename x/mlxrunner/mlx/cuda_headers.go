package mlx

import (
	"os"
	"path/filepath"
)

func CudaHeadersAvailable() bool {
	return cudaHeadersAvailable()
}

func cudaHeadersAvailable() bool {
	candidates := []string{}
	for _, envName := range []string{"CUDA_HOME", "CUDA_PATH"} {
		if dir := os.Getenv(envName); dir != "" {
			candidates = append(candidates, filepath.Join(dir, "include", "cuda", "std", "type_traits"))
		}
	}
	candidates = append(candidates,
		"/usr/local/cuda/include/cuda/std/type_traits",
		"/usr/include/cuda/std/type_traits",
	)
	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return true
		}
	}
	return false
}
