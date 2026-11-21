package main

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
