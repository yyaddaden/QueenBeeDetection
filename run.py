# -*- coding: UTF-8 -*-

from QueenBeeDetection import *

from argparse import ArgumentParser

if __name__ == "__main__":
    #####

    # parameters management
    args_parser = ArgumentParser(
        description="Queen Bee Detection and Recognition by Yacine YADDADEN [ https://github.com/yyaddaden ]"
    )
    args_parser.version = "1.0"

    group = args_parser.add_argument_group("train a model")
    group.add_argument("-t", "--train", help="training", action="store_true")
    group.add_argument(
        "-f", "--folder", metavar="", help="traning folder", action="store"
    )
    group.add_argument(
        "-c", "--components", metavar="", help="number of components", action="store"
    )

    group = args_parser.add_argument_group("perform recognition")
    group.add_argument("-r", "--recognition", help="recognition", action="store_true")
    group.add_argument("-i", "--image", metavar="", help="bee image", action="store")

    group = args_parser.add_argument_group("model evaluation")
    group.add_argument("-e", "--evaluation", help="evaluation", action="store_true")
    group.add_argument(
        "-df", "--datafolder", metavar="", help="dataset folder", action="store"
    )

    args = args_parser.parse_args()

    # parameters validation & execution
    queenBeeDetection = QueenBeeDetection()

    if args.train:
        queenBeeDetection.train_model(args.folder, True, int(args.components))
    elif args.recognition:
        queenBeeDetection.recognition(args.image)
    elif args.evaluation:
        queenBeeDetection.model_evaluation(args.datafolder, True)
    else:
        args_parser.print_help()
