import argparse
from src.parameters import AtariParameters
from src.setup import setup_for_evaluation

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, help="The model to be evaluated.")
parser.add_argument("--video_name",default=None, type=str, help="If set, records video and saves in videos/video_name")

args = parser.parse_args()
params = AtariParameters.from_model_name(args.model_name)

agent, runner = setup_for_evaluation(params,epsilon=0.05)
agent.load_network(args.model_name)

runner.run_test(render=True,video_folder_name=args.video_name)