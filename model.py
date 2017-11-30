import numpy as np

from uuid import uuid4


class Citizen:

    def __init__(self, num_concepts, num_features, vote_allotment=1.0):
        self.concepts = np.random.rand(num_concepts, num_features)

    def connect_with_elites(self, elites):
        pass

    def calculate_payoff(self, elites):
        pass

    def votes(self, elites):
        '''
        Determines distribution of votes from a single citizen.
        '''
        # Calculate softmax normalization factor.
        normalization = 1.0 / np.sum(
            np.exp(self.concepts.dot(elite.concepts)) for elite in elites
        )

        # Multiply softmax numerators for each elite and scale by norm factor.
        return (
            self.vote_allotment
            * normalization
            * np.array([
                np.exp(self.concepts.dot(elite.concepts))
                for elite in elites
            ])
        )

    def update_concepts(self, other_citizens):
        _update_concepts(self, other_citizens)


class Elite:

    def __init__(self, num_concepts, num_features=None, learning_rate=1.0):
        self.id = uuid4()
        if not num_features:
            self.concepts = np.random.rand(num_concepts)
        else:
            raise NotImplementedError('only 1D concept vectors currently!')

    def calculate_payoff(self, citizens):
        self.last_payoff = citizens.votes(self.id)

    def update_concepts(self, other_elites):
        _update_concepts(self, other_elites)


def _update_concepts(self, conspecifics):
    '''

    '''
    others_con_pay = ((o.concepts(), o.payoff()) for o in conspecifics)
    normalizers = (
        1.0 / (np.linalg.norm(self.concepts) * np.linalg.norm(o.concepts))
        for o in conspecifics
    )

    update = np.sum(
        pcn[0][0].dot(self.concepts)
        * pcn[1]
        * np.abs(self.payoff - pcn[0][1])

        for pcn in zip(others_con_pay, normalizers)
    )

    self.concepts += update


class Citizens:

    def __init__(self, elites):
        '''
        Container for all citizens
        '''
        pass

    @classmethod
    def create(cls, n_elites, n_concepts):
        elites = [Elite(n_concepts) for _ in n_elites]
        return cls(elites)


class Elites:


    def __init__(self, elites):
        '''
        Container for all elites
        '''
        self.elites = elites

    @classmethod
    def create(cls, n_elites, n_concepts):
        elites = [Elite(n_concepts) for _ in n_elites]
        return cls(elites)


class Simluation:

    def __init__(self, n_citizens=10, n_elites=5,
                 n_concepts=5, learning_rate=1.0):
        '''
        Create a new simulation
        '''
        self.elites = Elites.create(n_elites, n_concepts)
        self.citizens = Citizens.create(n_citizens, n_concepts)

    def citizens_vote(self):
        '''
        Each citizen observes elites, then votes based on coherence between
        the citizen's concepts and the
        '''
        # Build citizen-elite vote matrix.
        votes = np.array([
            citizen.votes(self.elites) for citizen in self.citizens
        ])
        for idx, elite in self.elites:
            elite.last_payoff = votes[idx].sum()

    def citizens_update(self):
        '''
        Citizens check other citizens' payoffs, then update their
        concepts based on observed payoffs.
        '''
        pass

    def elite_update(self):
        '''
        Based on number of votes collected, elites compute their payoff.
        Elites then observe other elites' payoffs and update theirs
        based on observed payoffs.
        '''
        pass

    def run(self, steps=1000):

        for t in steps:
            self.citizens_vote()
            self.elites_update()
            self.citizens_update()

def _agents_update(agents):
    '''
    Whether citizens or elites, inspect conspecifics' payoffs and update
    accordingly.
    '''
    pass
