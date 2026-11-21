package main

import (
	"errors"
	"fmt"
)

type gpu struct {
	name      string
	vram      float64
	bandwidth float64
}

var gpus = []gpu{
	{"h100-sxm", 80, 3350},
	{"h100-pcie", 80, 2000},
	{"h200-sxm", 141, 4800},
	{"a100-80", 80, 1935},
	{"l4", 24, 300},
}

func selectGPU(name string, vram, bandwidth float64, custom bool) (gpu, error) {
	if custom {
		if !validGB(vram) || !validGB(bandwidth) || vram == 0 || bandwidth == 0 {
			return gpu{}, errors.New("custom GPU requires positive vram and bandwidth")
		}
		return gpu{"custom", vram, bandwidth}, nil
	}
	for _, g := range gpus {
		if g.name == name {
			return g, nil
		}
	}
	return gpu{}, fmt.Errorf("unknown GPU %q", name)
}
