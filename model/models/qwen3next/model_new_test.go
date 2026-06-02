package qwen3next

import (
	"slices"
	"strings"
	"testing"
)

func TestInferRecurrentLayersMixedKVArray(t *testing.T) {
	got, err := inferRecurrentLayers([]uint64{0, 2, 0, 2}, 4, 0, 0)
	if err != nil {
		t.Fatalf("inferRecurrentLayers() error = %v", err)
	}

	want := []bool{true, false, true, false}
	if !slices.Equal(got, want) {
		t.Fatalf("inferRecurrentLayers() = %v, want %v", got, want)
	}
}

func TestInferRecurrentLayersScalarKVDefaultInterval(t *testing.T) {
	got, err := inferRecurrentLayers([]uint64{2, 2, 2, 2, 2, 2, 2, 2}, 8, 0, 0)
	if err != nil {
		t.Fatalf("inferRecurrentLayers() error = %v", err)
	}

	want := []bool{true, true, true, false, true, true, true, false}
	if !slices.Equal(got, want) {
		t.Fatalf("inferRecurrentLayers() = %v, want %v", got, want)
	}
}

func TestInferRecurrentLayersScalarKVConfiguredInterval(t *testing.T) {
	got, err := inferRecurrentLayers([]uint64{2, 2, 2, 2, 2, 2}, 6, 3, 0)
	if err != nil {
		t.Fatalf("inferRecurrentLayers() error = %v", err)
	}

	want := []bool{true, true, false, true, true, false}
	if !slices.Equal(got, want) {
		t.Fatalf("inferRecurrentLayers() = %v, want %v", got, want)
	}
}

func TestInferRecurrentLayersAllZeroRejects(t *testing.T) {
	_, err := inferRecurrentLayers([]uint64{0, 0, 0, 0}, 4, 0, 0)
	if err == nil {
		t.Fatal("inferRecurrentLayers() expected error, got nil")
	}
	if !strings.Contains(err.Error(), "must include at least one non-zero value") {
		t.Fatalf("unexpected error = %v", err)
	}
}

func TestInferRecurrentLayersWithMTP(t *testing.T) {
	// 32 base layers + 1 MTP layer = 33 layers
	// full_attention_interval = 4
	// nextn_predict_layers = 1
	// Using scalar KV count:
	got, err := inferRecurrentLayers([]uint64{4}, 33, 4, 1)
	if err != nil {
		t.Fatalf("inferRecurrentLayers() error = %v", err)
	}

	// First 32 layers follow the hybrid layout.
	// Last layer (layer 32, which is index 32) should be false (full attention).
	want := make([]bool, 33)
	for i := range 32 {
		want[i] = (i+1)%4 != 0
	}
	want[32] = false

	if !slices.Equal(got, want) {
		t.Fatalf("inferRecurrentLayers() = %v, want %v", got, want)
	}
}

func TestDefaultVHeadReordered(t *testing.T) {
	if !defaultVHeadReordered("qwen35") {
		t.Fatal("defaultVHeadReordered(qwen35) = false, want true")
	}
	if !defaultVHeadReordered("qwen35moe") {
		t.Fatal("defaultVHeadReordered(qwen35moe) = false, want true")
	}
	if defaultVHeadReordered("qwen3next") {
		t.Fatal("defaultVHeadReordered(qwen3next) = true, want false")
	}
}
