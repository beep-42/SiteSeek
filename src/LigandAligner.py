from Bio.PDB import PDBParser, PDBIO, StructureBuilder
from deap import base
from deap import tools
from deap import algorithms
from deap import creator
import random
from Bio.PDB.Superimposer import SVDSuperimposer
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


class LigandAligner:

    """
    This class uses evolutionary algorithm to find optimal ligand mapping, that minimizes element collisions
    (mismatches) and results in the lowest RMSD.
    """

    def __init__(self, path1, path2):

        """
        Initialise the LigandAligner
        :param path1: path to the first PDB file
        :param path2: path to the second PDB file
        """

        parser = PDBParser(PERMISSIVE=True, QUIET=True)
        with open(path1, 'r') as file:
            self.struct1 = parser.get_structure('struct1', file)
        with open(path2, 'r') as file:
            self.struct2 = parser.get_structure('struct2', file)

        self.switched = False
        if len(self.struct1) > len(self.struct2):
            self.struct1, self.struct2 = self.struct2, self.struct1
            self.switched = True

        self.atoms1 = list(self.struct1.get_atoms())
        self.atoms2 = list(self.struct2.get_atoms())

        self.coords1 = np.array([x.get_coord() for x in self.atoms1])
        self.coords2 = np.array([x.get_coord() for x in self.atoms2])

        self.elements1 = [x.element for x in self.atoms1]
        self.elements2 = [x.element for x in self.atoms2]

        self.length = min(len(self.atoms1), len(self.atoms2))

        self.rot_trans = None

        # print(f"Struct 1 has {len(self.atoms1)} atoms, struct 2 has {len(self.atoms2)} atoms.")

    def align_fast(self, fig: bool = False):
        """
        Runs the mapper and aligner.
        :param fig: Whether to show min fittness figure. Default is False
        :return: rmsd, rot, trans matrices, number of atom type mismatches in the final mapping
        """

        sup = SVDSuperimposer()

        # avoid multiple allocations
        x = np.zeros((self.length, 3))
        y = np.zeros((self.length, 3))

        def fitness(ind):
            nonlocal sup, x, y

            mismatches = 0
            c = 0

            for i, one in enumerate(ind):

                if i >= len(self.atoms1) or one >= len(self.atoms2): continue

                if self.atoms1[i].element != self.atoms2[one].element:
                    mismatches += 1

                x[c] = self.coords1[i]
                y[c] = self.coords2[one]

                c += 1

            sup.set(np.array(x), np.array(y))
            sup.run()

            # # Closest mapping experiment
            # ro, tran = sup.get_rotran()
            # coords = self.coords2 @ ro + tran
            #
            # for i, pos in enumerate(self.coords1):
            #     closest = np.argmin(np.linalg.norm(coords - pos, axis=1))
            #     x[i] = pos
            #     y[i] = self.coords2[closest]
            #
            # sup.set(np.array(x), np.array(y))
            # sup.run()

            return sup.get_rms() + mismatches*10,

        def get_rms_rot_trans(mapping):
            nonlocal sup

            x = []
            y = []
            mismatches = 0
            for i, one in enumerate(mapping):

                if i >= len(self.atoms1) or one >= len(self.atoms2): continue

                x.append(self.atoms1[i].get_coord())
                y.append(self.atoms2[one].get_coord())

                if self.atoms1[i].element != self.atoms2[one].element:

                    print("ELEMENT MISMATCH!")
                    print(self.atoms1[i].get_id())
                    print(self.atoms2[one].get_id())
                    mismatches += 1

            sup.set(np.array(x), np.array(y))
            sup.run()

            # Closest mapping experiment
            ro, tran = sup.get_rotran()
            coords = self.coords2 @ ro + tran

            closest_mismatches = 0

            p = 0
            for i, one in enumerate(mapping):
                if i >= len(self.atoms1) or one >= len(self.atoms2): continue

                closest = np.argmin(np.linalg.norm(coords - self.coords1[i], axis=1))
                x[p] = self.coords1[i]
                y[p] = self.coords2[closest]

                if self.elements1[i] != self.elements2[closest]:
                    closest_mismatches += 1

                p += 1

            sup.set(np.array(x), np.array(y))
            sup.run()

            print(f"Original mismatches: {mismatches} and new closest atoms mismatches: {closest_mismatches}")

            return sup.get_rms(), *sup.get_rotran(), mismatches

        SIZE = max(len(self.atoms1), len(self.atoms2))

        def get_heuristic_individual():
            ind = [-1] * SIZE
            for i in range(SIZE):
                if i < self.length:
                    at_id = self.atoms1[i].get_id()
                    for j in range(len(self.atoms2)):
                        if self.atoms2[j].get_id() == at_id:
                            ind[i] = j
                            break

            # replace -1s
            for i in range(SIZE):
                if ind[i] == -1:
                    for j in range(SIZE):
                        if j not in ind:
                            ind[i] = j
                            break

            return ind

        heuristic_ind = get_heuristic_individual()

        def perturb_heuristic_individual(indpb=0.05, rand_pb=.75):
            ind = heuristic_ind.copy()
            if random.random() > rand_pb:
                ind = tools.mutShuffleIndexes(ind, indpb=indpb)[0]
            else:
                return random.sample(range(SIZE), SIZE)

            return ind

        # toolbox = base.Toolbox()

        creator.create("FitnessMinRmsd", base.Fitness, weights=(-1.0,))     # fitness for minimization - notice the negative weight
        creator.create("Individual", list, fitness=creator.FitnessMinRmsd)  # create class for individuals - derived from list and uses the fitness defined above

        toolboxAlign = base.Toolbox()
        toolboxAlign.register("indices", perturb_heuristic_individual)
        toolboxAlign.register("individual", tools.initIterate, creator.Individual, toolboxAlign.indices) # initIterate can be used to initialize individual from any iterable object
        toolboxAlign.register("population", tools.initRepeat, list, toolboxAlign.individual)

        # print("indices:", toolboxAlign.indices())
        # print("individual:", toolboxAlign.individual())


        toolboxAlign.register("evaluate", fitness)
        toolboxAlign.register("mate", tools.cxOrdered)
        toolboxAlign.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        toolboxAlign.register("select", tools.selTournament, tournsize=5)

        pop = toolboxAlign.population(n=20)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        if fig:
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

        pop, log = algorithms.eaMuPlusLambda(pop, toolboxAlign, mu=20, lambda_=20, cxpb=0.3, mutpb=0.7, ngen=150,
                                             stats=stats, halloffame=hof, verbose=False)


        if fig:
            plt.plot([x['avg'] for x in log])
            plt.plot([x['min'] for x in log])

            plt.show()

        rms, rot, trans, mismatches = get_rms_rot_trans(hof[0])    # return rmsd, rot, trans of best individual
        self.rot_trans = (rot, trans)

        del creator.Individual, creator.FitnessMinRmsd

        return rms, rot, trans, mismatches

    def align(self, fig: bool = False, max_gen: int = 500, early_stopping: bool = False, look_behind: int = 30):
        """
        Runs the mapper and aligner.
        :param fig: Whether to show min fittness figure. Default is False
        :param early_stopping: Whether to stop if min score did not improve in the last look_behind generations.
        Default is False.
        :param look_behind: Number of generations to look behind when early_stopping is True.
        :return: rmsd, rot, trans matrices, number of atom type mismatches in the final mapping
        """

        sup = SVDSuperimposer()

        # avoid multiple allocations
        x = np.zeros((self.length, 3))
        y = np.zeros((self.length, 3))

        def fitness(ind):
            nonlocal sup, x, y

            mismatches = 0
            c = 0

            for i, one in enumerate(ind):

                if i >= len(self.atoms1) or one >= len(self.atoms2): continue

                if self.atoms1[i].element != self.atoms2[one].element:
                    mismatches += 1

                x[c] = self.coords1[i]
                y[c] = self.coords2[one]

                c += 1

            sup.set(np.array(x), np.array(y))
            sup.run()

            # # Closest mapping experiment
            # ro, tran = sup.get_rotran()
            # coords = self.coords2 @ ro + tran
            #
            # for i, pos in enumerate(self.coords1):
            #     closest = np.argmin(np.linalg.norm(coords - pos, axis=1))
            #     x[i] = pos
            #     y[i] = self.coords2[closest]
            #
            # sup.set(np.array(x), np.array(y))
            # sup.run()

            return sup.get_rms() + mismatches * 10,

        def get_rms_rot_trans(mapping):
            nonlocal sup

            x = []
            y = []
            mismatches = 0
            for i, one in enumerate(mapping):

                if i >= len(self.atoms1) or one >= len(self.atoms2): continue

                x.append(self.atoms1[i].get_coord())
                y.append(self.atoms2[one].get_coord())

                if self.atoms1[i].element != self.atoms2[one].element:
                    # print("ELEMENT MISMATCH!")
                    # print(self.atoms1[i].get_id())
                    # print(self.atoms2[one].get_id())
                    mismatches += 1

            sup.set(np.array(x), np.array(y))
            sup.run()

            return sup.get_rms(), *sup.get_rotran(), mismatches

        SIZE = max(len(self.atoms1), len(self.atoms2))

        # toolbox = base.Toolbox()

        creator.create("FitnessMinRmsd", base.Fitness,
                       weights=(-1.0,))  # fitness for minimization - notice the negative weight
        creator.create("Individual", list,
                       fitness=creator.FitnessMinRmsd)  # create class for individuals - derived from list and uses the fitness defined above

        toolboxAlign = base.Toolbox()
        toolboxAlign.register("indices", random.sample, range(SIZE), SIZE)
        toolboxAlign.register("individual", tools.initIterate, creator.Individual,
                              toolboxAlign.indices)  # initIterate can be used to initialize individual from any iterable object
        toolboxAlign.register("population", tools.initRepeat, list, toolboxAlign.individual)

        # print("indices:", toolboxAlign.indices())
        # print("individual:", toolboxAlign.individual())

        toolboxAlign.register("evaluate", fitness)
        toolboxAlign.register("mate", tools.cxOrdered)
        toolboxAlign.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        toolboxAlign.register("select", tools.selTournament, tournsize=5)

        pop = toolboxAlign.population(n=100)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        if fig:
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("max", np.max)

        if early_stopping:
            # run on 100 generations, then check improvement in the last look_behind generations
            pop, log = algorithms.eaMuPlusLambda(pop, toolboxAlign, mu=100, lambda_=100, cxpb=0.3, mutpb=0.7, ngen=look_behind,
                                                 stats=stats, halloffame=hof, verbose=False)
            done = look_behind

            last_section = [x['min'] for x in log[-look_behind:]]
            while min(last_section) != max(last_section) and done < max_gen:
                # only perform look_behind gens
                pop, log_cont = algorithms.eaMuPlusLambda(pop, toolboxAlign, mu=100, lambda_=100, cxpb=0.3, mutpb=0.7,
                                                     ngen=look_behind,
                                                     stats=stats, halloffame=hof, verbose=False)
                log += log_cont
                done += look_behind

        else:
            pop, log = algorithms.eaMuPlusLambda(pop, toolboxAlign, mu=100, lambda_=100, cxpb=0.3, mutpb=0.7,
                                                      ngen=max_gen,
                                                      stats=stats, halloffame=hof, verbose=False)

        if fig:
            plt.plot([x['avg'] for x in log])
            plt.plot([x['min'] for x in log])

            plt.show()

        rms, rot, trans, mismatches = get_rms_rot_trans(hof[0])  # return rmsd, rot, trans of best individual
        self.rot_trans = (rot, trans)

        del creator.Individual, creator.FitnessMinRmsd

        return rms, rot, trans, mismatches

    def align_continuously(self, fig: bool = False):
        """
        Performs alignment using finding the least sum of closest distances for evolving rot, trans.
        :param fig: Whether to show min fittness figure. Default is False
        :return: rmsd, rot, trans matrices
        """

        def fitness(ind):
            # sum of argmin of dists + # mismatches
            rot = R.from_rotvec(ind[:3])
            trans = ind[3:]

            all = []
            taken = np.zeros((self.length))
            for pos in self.coords1:
                all.append(
                    np.linalg.norm(
                        rot.apply(self.coords2) + trans - pos
                    , axis=1) + taken
                )
                taken[np.argmin(all[-1])] += 10

            s = np.sum(np.min(all, axis=1))
            mapping = np.argmin(all, axis=1)
            mismatches = 0
            for i in range(len(mapping)):
                if self.elements1[i] != self.elements2[mapping[i]]:
                    mismatches += 1

            # print(s)
            return s + mismatches,

        def rand_rot():

            return np.random.uniform(-2*np.pi, 2*np.pi, size=(1, 3))

        center = np.array(self.struct1.center_of_mass() - self.struct2.center_of_mass())
        disperse = 0.0

        def rand_trans():
            return np.add(disperse * np.random.standard_normal(size=(1, 3)), center)

        def rand_ind():

            return np.concatenate([rand_rot(), rand_trans()], axis=1).tolist()[0]

        def get_rms_rot_trans(ind):
            rot = R.from_rotvec(ind[:3])
            trans = ind[3:]

            all = []
            for pos in self.coords1:
                all.append(
                    np.linalg.norm(
                        rot.apply(self.coords2) + trans - pos
                        , axis=1)
                )
            mapping = np.argmin(all, axis=1)

            x = []
            y = []
            for i, one in enumerate(mapping):
                x.append(self.coords1[i])
                y.append(self.coords2[one])

            sup = SVDSuperimposer()
            sup.set(np.array(x), np.array(y)) if not self.switched else sup.set(np.array(y), np.array(x))
            sup.run()

            return sup.get_rms(), *sup.get_rotran()


        print(rand_ind())

        # toolbox = base.Toolbox()

        creator.create("FitnessMinSum", base.Fitness, weights=(-1.0,))     # fitness for minimization - notice the negative weight
        creator.create("Individual", list, fitness=creator.FitnessMinSum)  # create class for individuals - derived from list and uses the fitness defined above

        toolboxAlign = base.Toolbox()
        toolboxAlign.register("values", rand_ind)
        toolboxAlign.register("individual", tools.initIterate, creator.Individual, toolboxAlign.values) # initIterate can be used to initialize individual from any iterable object
        toolboxAlign.register("population", tools.initRepeat, list, toolboxAlign.individual)

        # print("indices:", toolboxAlign.indices())
        # print("individual:", toolboxAlign.individual())


        toolboxAlign.register("evaluate", fitness)
        toolboxAlign.register("mate", tools.cxTwoPoint)
        toolboxAlign.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.5)
        toolboxAlign.register("select", tools.selTournament, tournsize=10)

        pop = toolboxAlign.population(n=100)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        if fig:
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

        pop, log = algorithms.eaMuPlusLambda(pop, toolboxAlign, mu=100, lambda_=100, cxpb=0.2, mutpb=0.8, ngen=500,
                                             stats=stats, halloffame=hof, verbose=False)

        if fig:
            plt.plot([x['min'] for x in log])
            plt.show()

        rms, rot, trans = get_rms_rot_trans(hof[0])    # return rmsd, rot, trans of best individual
        self.rot_trans = (rot, trans)
        print(rms)

        del creator.Individual, creator.FitnessMinSum
        return 0,0,0
        #return rms, rot, trans


    def save_superimposed(self, target_file, include_proteins = False):

        """
        Save the superimposed ligands into a given file. The aligner needs to be run first.
        :param target_file: target to the PDB output file.
        :return:
        """
        if self.rot_trans is None:
            raise Exception("Run the alignment first!")

        # struct = self.struct1.copy()
        # list(struct.get_chains())[0].id = 'A'
        # chain = list(self.struct2.get_chains())[0]
        # chain.id = 'B'
        # for atom in chain:
        #     atom.transform(self.rot_trans[0], self.rot_trans[1])
        #
        # chain.detach_parent()
        # chain.serial_num = 0
        # struct.add(chain)
        #
        # pdb_io = PDBIO()
        # pdb_io.set_structure(struct)
        # with open(target_file, 'w') as file:
        #     pdb_io.save(file)

        io = PDBIO()

        struct2 = self.struct2.copy()
        for atom in struct2.get_atoms():
            atom.transform(self.rot_trans[0], self.rot_trans[1])

        with open(target_file, 'w') as file:
            for struct in [self.struct1, struct2]:
                io.set_structure(struct)
                io.save(file)


if __name__ == "__main__":
    la = LigandAligner('../../TOUGH M1/TOUGH-M1_dataset/1lkxD/1lkxD00.pdb', '../../TOUGH M1/TOUGH-M1_dataset/1ii7A/1ii7A00.pdb')
    rmsd, rot, trans = la.align(True)
    print(f'Reached rmsd: {rmsd}')
    la.save_superimposed('../../TOUGH M1/superimposed-ligands.pdb')
