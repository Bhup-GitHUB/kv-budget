package main

import "math"

const gb = 1e9

type options struct {
	weightBytes float64
	kvBytes     float64
	reserveGB   float64
	prefix      int
	noGQA       bool
}

type budget struct {
	kvPerToken float64
	kvPerUser  float64
	sharedKV   float64
	users      float64
	speed      float64
}

func calculate(m model, g gpu, context int, opts options) budget {
	heads := m.kvHeads
	if opts.noGQA {
		heads = m.heads
	}
	kvPerToken := 2 * float64(m.layers) * float64(heads) * float64(m.headDim) * opts.kvBytes
	kvPerUser := kvPerToken * float64(context-opts.prefix)
	sharedKV := kvPerToken * float64(opts.prefix)
	weights := m.params * opts.weightBytes
	free := g.vram*gb - weights - opts.reserveGB*gb - sharedKV
	speed := g.bandwidth * gb / (weights + kvPerToken*float64(context))
	return budget{kvPerToken, kvPerUser, sharedKV, usersThatFit(free, kvPerUser), speed}
}

func usersThatFit(free, kvPerUser float64) float64 {
	if free < 0 {
		return 0
	}
	if kvPerUser == 0 {
		return math.Inf(1)
	}
	return math.Floor(free / kvPerUser)
}
