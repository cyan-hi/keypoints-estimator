from ultralytics import YOLO
import logging

from utils import utils
from tracker.deep_sort import DeepSort
from model.keypoint_estimator import KeypointEstimator
from pipeline.stream_runner import StreamRunner

utils.setting_logger()
logger = logging.getLogger(__name__)

class Main:
    def __init__(self):
        self.config = utils.load_config()
        logger.info(f"Generate Deep Learning Model")

        try:
            self.detector = YOLO(self.config["detector"]["weight"])
            logger.info(f'Loading YOLO weight from {self.config["detector"]["weight"]}... Done!')

            self.tracker = DeepSort(self.config["tracker"])
            logger.info(f'Loading DeepSort weight from {self.config["tracker"]["reid_checkpoint"]}... Done!')

            self.keypoint_estimater = KeypointEstimator(self.config["keypoint_estimator"]["weight"])
            logger.info(f'Loading RTMPose weight from {self.config["keypoint_estimator"]["weight"]}... Done!')

            self.stream_runner = StreamRunner(self.config, self.detector, self.tracker, self.keypoint_estimater)
            logger.info(f"Success to Load Stream Runner")

        except Exception as e:
            logger.error(f"Fail to Load Model: {e}", exc_info=True)

        self.start()

    def start(self):
        logger.info(f"Start Main Process ...")
        self.stream_runner.run()


if __name__ == "__main__":
    main = Main()
