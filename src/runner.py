from argparse import ArgumentParser
from pathlib import Path

from .profiler import main_iris, main_digits, iris_fitting, digits_fitting, draw_histogram


def _positive(value):
    value = int(value)
    if value < 1:
        from argparse import ArgumentTypeError
        raise ArgumentTypeError("must be positive")
    return value


def main():
    parser = ArgumentParser(description="ADMM classifier benchmark")
    parser.add_argument("dataset", choices=("iris", "digits"))
    parser.add_argument("--repetitions", type=_positive, default=10)
    parser.add_argument("--iterations", type=_positive, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", choices=("curve", "histogram"),
                        help="show a learning curve or benchmark accuracy histogram")
    parser.add_argument("--output", type=Path, metavar="PATH",
                        help="save plot instead of opening a window (PNG, PDF, SVG, ...)")
    args = parser.parse_args()
    if args.output is not None and args.plot is None:
        parser.error("--output requires --plot")

    if args.plot:
        try:
            import matplotlib
        except ModuleNotFoundError as error:
            if error.name != "matplotlib":
                raise
            parser.error("plotting requires Matplotlib: python -m pip install -e '.[plot]'")
        if args.output is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if args.plot == "curve":
            fitting = iris_fitting if args.dataset == "iris" else digits_fitting
            fig = fitting(m=args.repetitions, k=args.iterations, rng=args.seed)
        else:
            benchmark = main_iris if args.dataset == "iris" else main_digits
            results = benchmark(args.repetitions, args.iterations, args.seed)
            fig = draw_histogram([result.accuracy for result in results], args.dataset)
        try:
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(args.output, bbox_inches="tight")
                print(f"Saved plot to {args.output}")
            else:
                plt.show()
        finally:
            plt.close(fig)
        return

    benchmark = main_iris if args.dataset == "iris" else main_digits
    benchmark(args.repetitions, args.iterations, args.seed)
