package main

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestBudget(t *testing.T) {
	m := models["llama-3.1-8b"]
	for _, tc := range []struct {
		name    string
		gpu     gpu
		context int
		opts    options
		token   float64
		userGB  float64
		users   float64
		speed   float64
	}{
		{"full context", gpus[0], 131072, options{weightBytes: 2, kvBytes: 2}, 131072, 17.179869184, 3, 100.78},
		{"8k", gpus[0], 8192, options{weightBytes: 2, kvBytes: 2}, 131072, 1.073741824, 59, 195.52},
		{"shared prefix", gpus[0], 8192, options{weightBytes: 2, kvBytes: 2, prefix: 4096}, 131072, 0.536870912, 118, 195.52},
		{"all shared", gpus[0], 8192, options{weightBytes: 2, kvBytes: 2, prefix: 8192}, 131072, 0, math.Inf(1), 195.52},
		{"no gqa", gpus[0], 8192, options{weightBytes: 2, kvBytes: 2, noGQA: true}, 524288, 4.294967296, 14, 164.58},
		{"int4 and fp8", gpus[4], 8192, options{weightBytes: 0.5, kvBytes: 1, reserveGB: 20}, 65536, 0.536870912, 0, 65.91},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := calculate(m, tc.gpu, tc.context, tc.opts)
			if got.kvPerToken != tc.token || math.Abs(got.kvPerUser/gb-tc.userGB) > 1e-9 || got.users != tc.users || math.Abs(got.speed-tc.speed) > 0.01 {
				t.Fatalf("got %+v, want %.0f bytes/token, %.9f GB/user, %.0f users, %.2f tokens/sec",
					got, tc.token, tc.userGB, tc.users, tc.speed)
			}
			if got.sharedKV != tc.token*float64(tc.opts.prefix) {
				t.Fatalf("got %.0f shared bytes", got.sharedKV)
			}
		})
	}
}

func TestLoadModel(t *testing.T) {
	for _, tc := range []struct {
		name   string
		config string
		params float64
		want   model
	}{
		{"llama parameters", `{"model_type":"llama","num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"hidden_size":4096,"max_position_embeddings":131072,"intermediate_size":14336,"vocab_size":128256}`, 0, model{8030261248, 32, 32, 8, 128, 131072}},
		{"explicit head dimension", `{"num_hidden_layers":2,"num_attention_heads":8,"head_dim":16,"max_position_embeddings":4096}`, 1234, model{1234, 2, 8, 8, 16, 4096}},
		{"invalid hidden size", `{"num_hidden_layers":2,"num_attention_heads":8,"hidden_size":63,"max_position_embeddings":4096}`, 1234, model{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.json")
			if err := os.WriteFile(path, []byte(tc.config), 0600); err != nil {
				t.Fatal(err)
			}
			got, err := loadModel(path, tc.params)
			if (err != nil) != (tc.want == (model{})) || got != tc.want {
				t.Fatalf("got %+v, %v; want %+v", got, err, tc.want)
			}
		})
	}
}
