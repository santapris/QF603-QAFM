"""
Generate Stage 4's figures: the period-level bias series (optionally with an
excluded window shaded, for a regime-sensitivity check), ACF/PACF of the
level series, residual diagnostics (ACF(residual), ACF(residual^2)) for the
full sample and optionally a sub-sample, and the Andrews sup-Wald
structural-break profile. Works against either a raw Kalshi pull
(--kalshi-path, harmonized here) or an already common-schema panel
(--panel-path, e.g. from pull_kalshi_ladder_atm.py) -- see README.md's
"Stage 4" section for how these figures were used on both KXTRUMPAPPROVE
(superseded -- see DECISIONS.md) and the current ladder-derived panel.

Usage:
  python scripts/plot_stage4.py --panel-path data/raw_kalshi_ladder_full.parquet \
      --prefix stage4_ladder_full --label "IL+TX gas price (full ladder)"
  python scripts/plot_stage4.py --kalshi-path data/raw_kalshi_trumpapprove.parquet \
      --exclude-before 2026-09-09 --prefix stage4_trumpapprove --label KXTRUMPAPPROVE
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.align import combine, harmonize_kalshi, load_panel  # noqa: E402
from src.models.stage4_persistence import (  # noqa: E402
    ar_structural_break_test,
    fit_ar_persistence,
    period_bias_series,
    plot_bias_series,
    plot_break_test_profile,
    plot_level_acf_pacf,
    plot_residual_diagnostics,
    select_ar_order,
)

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kalshi-path", default=None, help="raw Kalshi pull to harmonize, e.g. data/raw_kalshi_trumpapprove.parquet")
    parser.add_argument("--panel-path", default=None, help="already common-schema panel, e.g. from pull_kalshi_ladder_atm.py")
    parser.add_argument("--freq", default="D")
    parser.add_argument("--min-markets-per-period", type=int, default=1)
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--exclude-before", default=None,
                        help="ISO date: regenerate the residual-diagnostics figure on the "
                             "sub-sample from this date onward too, as a regime-sensitivity check")
    parser.add_argument("--out-dir", default="outputs/figures")
    parser.add_argument("--prefix", default="stage4")
    parser.add_argument("--label", default="bias series", help="human-readable series name for plot titles")
    parser.add_argument("--break-trim", type=float, default=0.15)
    parser.add_argument("--break-boot-reps", type=int, default=999)
    args = parser.parse_args()

    if not args.kalshi_path and not args.panel_path:
        parser.error("pass --kalshi-path or --panel-path")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.panel_path:
        panel = load_panel(ROOT / args.panel_path)
    else:
        panel = combine(harmonize_kalshi(pd.read_parquet(ROOT / args.kalshi_path)))
    periods = period_bias_series(panel, freq=args.freq, min_markets_per_period=args.min_markets_per_period)
    x = periods.set_index("period")["bias"]

    exclude_before = pd.Timestamp(args.exclude_before) if args.exclude_before else None

    bias_path = out_dir / f"{args.prefix}_bias_series.png"
    plot_bias_series(periods, bias_path, exclude_before=exclude_before,
                     title=f"{args.label} daily bias (freq={args.freq})")
    print(f"wrote {bias_path}")

    acf_path = out_dir / f"{args.prefix}_level_acf_pacf.png"
    plot_level_acf_pacf(x, acf_path, title=f"{args.label}: level bias series (full sample)")
    print(f"wrote {acf_path}")

    p_full = select_ar_order(x, max_p=args.max_p)
    fit_full = fit_ar_persistence(x, p_full)
    full_path = out_dir / f"{args.prefix}_residual_diagnostics_full.png"
    plot_residual_diagnostics(fit_full["resid"], full_path,
                              title=f"{args.label}: AR({p_full}) on full sample (n={len(x)}, "
                                    f"phi={fit_full['coefs'].get('lag1', float('nan')):.2f})")
    print(f"wrote {full_path}")

    if exclude_before is not None:
        x_sub = x[x.index >= exclude_before]
        p_sub = select_ar_order(x_sub, max_p=args.max_p)
        fit_sub = fit_ar_persistence(x_sub, p_sub)
        sub_path = out_dir / f"{args.prefix}_residual_diagnostics_post_launch.png"
        plot_residual_diagnostics(fit_sub["resid"], sub_path,
                                  title=f"{args.label}: AR({p_sub}) excluding before {args.exclude_before} "
                                        f"(n={len(x_sub)}, phi={fit_sub['coefs'].get('lag1', float('nan')):.2f})")
        print(f"wrote {sub_path}")

    break_result = ar_structural_break_test(x, p=p_full, trim=args.break_trim,
                                            boot_reps=args.break_boot_reps)
    print(f"Andrews sup-Wald: stat={break_result['sup_wald']:.3f}  "
          f"break_date={break_result['break_date'].date()}  "
          f"p_bootstrap={break_result['p_value']:.3f}  p_asymptotic={break_result['p_asymptotic']:.4f}")
    break_path = out_dir / f"{args.prefix}_break_test_profile.png"
    plot_break_test_profile(break_result, break_path,
                            title=f"{args.label}: Andrews sup-Wald on AR({p_full}) coefficients "
                                  f"(bootstrap p={break_result['p_value']:.2f})")
    print(f"wrote {break_path}")
