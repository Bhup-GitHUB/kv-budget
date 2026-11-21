package main

type model struct {
	name    string
	params  float64
	layers  int
	heads   int
	kvHeads int
	headDim int
	maxCtx  int
}

var models = map[string]model{
	"llama-3.1-8b": {"llama-3.1-8b", 8.03e9, 32, 32, 8, 128, 131072},
}
