package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
)

type model struct {
	params  float64
	layers  int
	heads   int
	kvHeads int
	headDim int
	maxCtx  int
}

var models = map[string]model{
	"llama-3.1-8b": {8.03e9, 32, 32, 8, 128, 131072},
}

type hfConfig struct {
	ModelType        string  `json:"model_type"`
	Params           float64 `json:"num_parameters"`
	Layers           int     `json:"num_hidden_layers"`
	Heads            int     `json:"num_attention_heads"`
	KVHeads          int     `json:"num_key_value_heads"`
	HeadDim          int     `json:"head_dim"`
	MaxCtx           int     `json:"max_position_embeddings"`
	HiddenSize       int     `json:"hidden_size"`
	IntermediateSize int     `json:"intermediate_size"`
	VocabSize        int     `json:"vocab_size"`
	TieEmbeddings    bool    `json:"tie_word_embeddings"`
	AttentionBias    bool    `json:"attention_bias"`
	MLPBias          bool    `json:"mlp_bias"`
}

func selectModel(name, path string, params float64) (model, error) {
	if path != "" {
		m, err := loadModel(path, params)
		if err != nil {
			return model{}, fmt.Errorf("config: %w", err)
		}
		return m, nil
	}
	if params != 0 {
		return model{}, errors.New("params requires config")
	}
	m, ok := models[name]
	if !ok {
		return model{}, fmt.Errorf("unknown model %q", name)
	}
	return m, nil
}

func loadModel(path string, params float64) (model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return model{}, err
	}
	var c hfConfig
	if err := json.Unmarshal(data, &c); err != nil {
		return model{}, err
	}
	if c.Layers <= 0 || c.Heads <= 0 || c.MaxCtx <= 0 {
		return model{}, errors.New("expected positive num_hidden_layers, num_attention_heads, and max_position_embeddings")
	}
	if c.KVHeads == 0 {
		c.KVHeads = c.Heads
	}
	if c.KVHeads < 0 || c.Heads%c.KVHeads != 0 {
		return model{}, errors.New("num_key_value_heads must divide num_attention_heads")
	}
	if c.HeadDim == 0 && c.HiddenSize%c.Heads == 0 {
		c.HeadDim = c.HiddenSize / c.Heads
	}
	if c.HeadDim <= 0 {
		return model{}, errors.New("expected head_dim or hidden_size divisible by num_attention_heads")
	}
	if params == 0 {
		params = c.Params
	}
	if params == 0 {
		params = llamaParams(c)
	}
	if !(params > 0 && params <= math.MaxFloat64/4) {
		return model{}, errors.New("specify a positive parameter count with --params")
	}
	return model{params, c.Layers, c.Heads, c.KVHeads, c.HeadDim, c.MaxCtx}, nil
}

func llamaParams(c hfConfig) float64 {
	if c.ModelType != "llama" || c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.VocabSize <= 0 {
		return 0
	}
	hidden, intermediate := float64(c.HiddenSize), float64(c.IntermediateSize)
	query := float64(c.Heads) * float64(c.HeadDim)
	keyValue := float64(c.KVHeads) * float64(c.HeadDim)
	attention := 2 * hidden * (query + keyValue)
	mlp := 3 * hidden * intermediate
	if c.AttentionBias {
		attention += query + 2*keyValue + hidden
	}
	if c.MLPBias {
		mlp += 2*intermediate + hidden
	}
	embeddings := hidden * float64(c.VocabSize)
	if !c.TieEmbeddings {
		embeddings *= 2
	}
	return embeddings + float64(c.Layers)*(attention+mlp+2*hidden) + hidden
}
