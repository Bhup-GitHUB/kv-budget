package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"text/tabwriter"
)

type row struct {
	gpu     string
	context int
	budget  budget
}

func userCount(users float64) string {
	if math.IsInf(users, 1) {
		return "unlimited"
	}
	return strconv.FormatFloat(users, 'f', 0, 64)
}

func printFit(out io.Writer, b budget) error {
	_, err := fmt.Fprintf(out, "KV per token: %.0f bytes\nKV per user: %.3f GB\nShared KV: %.3f GB\nUsers: %s\nSpeed ceiling: %.2f tokens/sec\n",
		b.kvPerToken, b.kvPerUser/gb, b.sharedKV/gb, userCount(b.users), b.speed)
	return err
}

func printTable(out io.Writer, rows []row) error {
	w := tabwriter.NewWriter(out, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "GPU\tContext\tKV/user (GB)\tShared KV (GB)\tUsers\tCeiling (tokens/sec)")
	for _, r := range rows {
		b := r.budget
		fmt.Fprintf(w, "%s\t%d\t%.3f\t%.3f\t%s\t%.2f\n",
			r.gpu, r.context, b.kvPerUser/gb, b.sharedKV/gb, userCount(b.users), b.speed)
	}
	return w.Flush()
}

func sweep(out io.Writer, m model, g gpu, opts options) error {
	var rows []row
	for _, context := range []int{2048, 8192, 32768, 131072, m.maxCtx} {
		if context > m.maxCtx || context < opts.prefix || (len(rows) > 0 && rows[len(rows)-1].context == context) {
			continue
		}
		rows = append(rows, row{g.name, context, calculate(m, g, context, opts)})
	}
	if err := writeSweep(rows); err != nil {
		return fmt.Errorf("sweep.csv: %w", err)
	}
	return printTable(out, rows)
}

func writeSweep(rows []row) error {
	records := [][]string{{"gpu", "context_tokens", "kv_bytes_per_token", "kv_bytes_per_user", "shared_kv_bytes", "users", "speed_ceiling_tokens_per_sec"}}
	for _, r := range rows {
		b := r.budget
		records = append(records, []string{
			r.gpu, strconv.Itoa(r.context),
			fmt.Sprintf("%.0f", b.kvPerToken), fmt.Sprintf("%.0f", b.kvPerUser),
			fmt.Sprintf("%.0f", b.sharedKV), userCount(b.users), fmt.Sprintf("%.2f", b.speed),
		})
	}
	f, err := os.Create("sweep.csv")
	if err != nil {
		return err
	}
	err = csv.NewWriter(f).WriteAll(records)
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}
