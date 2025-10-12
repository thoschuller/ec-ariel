import constants
from terminal import console, progress
from utils import plot_saved_phenotype, plot_and_record_saved_phenotype

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Plot, record, or evolve phenotypes.")
    parser.add_argument("--evolve", choices=["body", "brain"], help="Evolve mode: 'body' or 'brain'.")
    parser.add_argument("--brain", type=str, help="Path to the brain .npy file (for plotting/recording).")
    parser.add_argument("--body", type=str, help="Path to the body .json file (for plotting/recording or brain evolution).")
    parser.add_argument("--weights", type=str, help="Path to the weights file (for brain evolution, optional).")
    parser.add_argument(
        "--duration", type=float, default=constants.STAGE_SETTINGS["FULL"]["DURATION"], help="Duration of the simulation."
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Record a video with a default filename (output.mp4).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="headless",
        help="Method for running the session (record, headless, etc.).",
    )

    args = parser.parse_args()
    
    sys.stdout.flush()
    progress.start()

    try:       
        
        console.rule("Starting Ministry of Silly Walks AI...")
        
        if args.evolve == "body":
            console.log("Running body evolution...")
            from body_evolver import body_evolution
            result = body_evolution()
            console.log(f"body_evolution() result: {result}")
        elif args.evolve == "brain":
            if not args.body:
                console.log("[red] [ERROR] --body must be set for --evolve brain mode.")
                sys.exit(1)
            console.log("Running brain evolution...")
            from brain_train import train_individual_from_files
            # weights is optional
            result = train_individual_from_files(body_file=args.body, weights_file=args.weights)
            console.log(f"result: {result}")
        else:
            if not args.brain or not args.body:
                console.log("[red] [ERROR] --brain and --body must be set for plotting/recording.")
                sys.exit(1)
            if args.video:
                plot_and_record_saved_phenotype(
                    brain_file=args.brain,
                    body_file=args.body,
                    duration=args.duration,
                    video_filename="output.mp4",
                )
                console.log("Recording and plotting completed.")
            else:
                plot_saved_phenotype(
                    brain_file=args.brain,
                    body_file=args.body,
                    duration=args.duration,
                    method=args.method,
                )
                console.log("Plotting completed.")
        sys.stdout.flush()
    except Exception as e:
        console.log(f"[red] [ERROR] Exception in main: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    finally:
        progress.stop()