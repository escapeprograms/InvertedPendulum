import pickle
import os
import neat
from YourControlCode import Genome
from TrainSimulation import launch_from_path
from Graph import plot_stats

# Get directory path
dir_path = os.path.dirname(os.path.realpath(__file__))

# Original and modified model paths
robot_model = os.path.join(dir_path, "./Robot/miniArm_with_pendulum.xml")

def eval_genomes(genomes, config):
    genome_list = []
    for genome_id, genome in genomes:
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        genome_list.append(Genome(genome_id, network))

    #run simulation
    launch_from_path(robot_model, genome_list)

    #update fitnesses
    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = genome_list[i].fitness


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    #get data for the graph
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 20 generations.
    winner = p.run(eval_genomes, 20)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)


    return winner_net, stats


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-config')
    winner, stats = run(config_path)

    #create graph of training
    plot_stats(stats, ylog=False, view=True, title="Basic Crutch Fitness", filename='basic_crutch_fitness.svg')

    #save model
    with open("neat-model 8.pkl", "wb") as f:
        pickle.dump(winner, f)
    