#pragma once

extern int LLAMA_BUILD_NUMBER;
extern const char *LLAMA_COMMIT;
extern const char *LLAMA_COMPILER;
extern const char *LLAMA_BUILD_TARGET;

inline int llama_build_number() {
	return LLAMA_BUILD_NUMBER;
}

inline const char * llama_commit() {
	return LLAMA_COMMIT;
}

inline const char * llama_compiler() {
	return LLAMA_COMPILER;
}

inline const char * llama_build_target() {
	return LLAMA_BUILD_TARGET;
}