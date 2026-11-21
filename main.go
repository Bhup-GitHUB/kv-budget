package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string, out io.Writer) error {
	if len(args) == 0 {
		return errors.New("usage: kv-budget fit|sweep|compare [flags]")
	}
	command := args[0]
	switch command {
	case "-h", "--help":
		fmt.Fprintln(out, "usage: kv-budget fit|sweep|compare [flags]")
		return nil
	case "fit", "sweep", "compare":
	default:
		return fmt.Errorf("unknown command %q", command)
	}

	flags := flag.NewFlagSet("kv-budget "+command, flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	modelName := flags.String("model", "llama-3.1-8b", "built-in model name")
	precision := flags.String("precision", "fp16", "weight precision (fp16, fp8, int4)")
	kvPrecision := flags.String("kv-precision", "fp16", "KV precision (fp16, fp8)")
	var opts options
	flags.Float64Var(&opts.reserveGB, "reserve-gb", 0, "VRAM to reserve in GB")
	flags.IntVar(&opts.prefix, "shared-prefix", 0, "tokens shared by all users")
	flags.BoolVar(&opts.noGQA, "no-gqa", false, "use all attention heads for KV")

	var context int
	if command != "sweep" {
		flags.IntVar(&context, "context", 8192, "context length in tokens")
	}

	var gpuName string
	var vram, bandwidth float64
	if command != "compare" {
		flags.StringVar(&gpuName, "gpu", "h100-sxm", "built-in GPU name")
		flags.Float64Var(&vram, "vram", 0, "custom GPU VRAM in GB")
		flags.Float64Var(&bandwidth, "bandwidth", 0, "custom GPU memory bandwidth in GB/s")
	}
	flags.Usage = func() {}
	if err := flags.Parse(args[1:]); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fmt.Fprintf(out, "usage: kv-budget %s [flags]\n", command)
			flags.SetOutput(out)
			flags.PrintDefaults()
			return nil
		}
		return err
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected argument %q", flags.Arg(0))
	}

	m, ok := models[*modelName]
	if !ok {
		return fmt.Errorf("unknown model %q", *modelName)
	}
	opts.weightBytes = bytesFor(*precision)
	if opts.weightBytes == 0 {
		return errors.New("precision must be fp16, fp8, or int4")
	}
	opts.kvBytes = bytesFor(*kvPrecision)
	if opts.kvBytes < 1 {
		return errors.New("kv-precision must be fp16 or fp8")
	}
	if !validGB(opts.reserveGB) {
		return errors.New("reserve-gb must be finite and nonnegative")
	}
	if command == "sweep" {
		context = m.maxCtx
	}
	if context < 1 || context > m.maxCtx {
		return fmt.Errorf("context must be between 1 and %d", m.maxCtx)
	}
	if opts.prefix < 0 || opts.prefix > context {
		return errors.New("shared-prefix must be between 0 and the context length")
	}

	if command == "compare" {
		var rows []row
		for _, g := range gpus {
			rows = append(rows, row{g.name, context, calculate(m, g, context, opts)})
		}
		return printTable(out, rows)
	}

	custom := false
	flags.Visit(func(f *flag.Flag) {
		if f.Name == "vram" || f.Name == "bandwidth" {
			custom = true
		}
	})
	g, err := selectGPU(gpuName, vram, bandwidth, custom)
	if err != nil {
		return err
	}
	if command == "fit" {
		return printFit(out, calculate(m, g, context, opts))
	}
	return sweep(out, m, g, opts)
}

func bytesFor(precision string) float64 {
	switch precision {
	case "fp16":
		return 2
	case "fp8":
		return 1
	case "int4":
		return 0.5
	default:
		return 0
	}
}

func validGB(value float64) bool {
	return value >= 0 && value <= math.MaxFloat64/gb
}
