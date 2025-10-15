import constants
from terminal import console, progress

if __name__ == "__main__":
    import argparse
    import sys

    default_duration = constants.STAGE_SETTINGS["FULL"]["DURATION"]

    parser = argparse.ArgumentParser(
        description="Plot, record, or evolve Silly Walkers."
    )
    parser.add_argument(
        "--evolve", choices=["body", "brain"], help="Evolve mode: 'body' or 'brain'."
    )
    parser.add_argument(
        "--brain",
        type=str,
        help="Path to the brain .npy file (for plotting/recording).",
    )
    parser.add_argument(
        "--body",
        type=str,
        help="Path to the body .json file (for plotting/recording or brain evolution).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=default_duration,
        help="Duration of the simulation.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Record a video with a default filename (output.mp4).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Plot the phenotype's path and fitness.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show the specified brain and body in a viewer.",
    )

    args = parser.parse_args()

    sys.stdout.flush()
    progress.start()

    status = 0

    try:

        console.rule("Starting Ministry of Silly Walks EA...")

        if [args.evolve, args.viewer, args.plot, args.video].count(True) > 1:
            console.log(
                "Only one of --evolve, --viewer, --plot, or --video can be set at a time."
            )
            status = 1

        if args.evolve or args.viewer:
            if args.duration != default_duration:
                console.log("[yellow] [WARNING] --duration is ignored in this mode.")

            if args.viewer:
                if not args.brain or not args.body:
                    console.log(
                        "[red] [ERROR] --brain and --body must be set for viewer mode."
                    )
                    sys.exit(1)
                console.log("Launching viewer...")
                from utils import run_saved_phenotype

                run_saved_phenotype(
                    brain_file=args.brain,
                    body_file=args.body,
                    method="viewer",
                )
                console.log("Viewer session ended.")

            elif args.evolve == "body":
                console.log("Running body evolution...")
                full_evolve_task = progress.add_task(
                    "Full Evolution Progress", total=None
                )
                from body_evolver import body_evolution
                if constants.RANDOM_BASELINE:
                    from fake_brain_train import train_individual_from_files
                else:
                    from brain_train import train_individual_from_files

                result = body_evolution()
                fit, _, brain, body_graph = result
                console.log(f"Body evolution returned fitness: {fit}")
                console.rule(f"Body evolution completed. Best fitness: {fit}")
                console.rule(f"Starting prolonged brain training on best body...")
                brain_result = train_individual_from_files(
                    body_graph=body_graph,
                    np_weights=brain,
                    time_limit=constants.EXTRA_TRAINING_TIME,
                    record_batch=constants.BRAIN_EVO_RECORD_BATCH,
                    record_last=constants.BRAIN_EVO_RECORD_LAST,
                )
                console.log(f"Prolonged brain training returned: {brain_result[1]}")
                console.rule(
                    f"Prolonged brain training completed. Best fitness: {brain_result[1]}"
                )
                progress.remove_task(full_evolve_task)

            elif args.evolve == "brain":
                if not args.body:
                    console.log(
                        "[red] [ERROR] --body must be set for --evolve brain mode."
                    )
                    sys.exit(1)
                console.log("Running brain evolution...")
                if constants.RANDOM_BASELINE:
                    from fake_brain_train import train_individual_from_files
                else:
                    from brain_train import train_individual_from_files

                # weights is optional
                result = train_individual_from_files(
                    body_file=args.body, weights_file=args.brain, record_batch=True
                )
                console.log(f"result: {result}")

        elif args.plot or args.video:
            if not args.brain or not args.body:
                console.log(
                    "[red] [ERROR] --brain and --body must be set for plotting/recording."
                )
                sys.exit(1)
            if args.video:
                console.log("Recording video and plotting path...")
                from utils import plot_and_record_saved_phenotype, plot_saved_phenotype

                plot_and_record_saved_phenotype(
                    brain_file=args.brain,
                    body_file=args.body,
                    duration=args.duration,
                    video_filename="output.mp4",
                )
                console.log("Recording and plotting completed.")
            else:
                console.log("Plotting path...")
                from utils import plot_saved_phenotype

                plot_saved_phenotype(
                    brain_file=args.brain,
                    body_file=args.body,
                    duration=args.duration,
                    method="headless",
                )
                console.log("Plotting completed.")
        else:
            # show help if no arguments are provided
            if len(sys.argv) == 1:
                parser.print_help(sys.stderr)
                status = 1
        sys.stdout.flush()
    except Exception as e:
        console.log(f"[red] [ERROR] Exception in Evolver: {e}")
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        status = 1
    finally:
        progress.stop()
        exit(status)
