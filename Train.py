import os
import neat
from YourControlCode import Genome
from TrainSimulation import launch_from_path

# Get directory path
dir_path = os.path.dirname(os.path.realpath(__file__))

# Original and modified model paths
robot_model = os.path.join(dir_path, "./Robot/miniArm_with_pendulum.xml")

def eval_genomes(genomes, config):
    genome_list = []
    for genome_id, genome in genomes:
        genome.fitness = 4.0
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
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 20 generations.
    winner = p.run(eval_genomes, 20)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-config')
    run(config_path)