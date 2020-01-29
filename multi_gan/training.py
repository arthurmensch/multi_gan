from sklearn.utils import check_random_state


class Scheduler:
    def __init__(self, n_generators, n_discriminators, random_state=None, shuffle=True, sampling='pair'):
        self.shuffle = shuffle
        self.sampling = sampling
        self.random_state = check_random_state(random_state)
        self.n_computations = 0
        self.i, self.j = 0, 0
        self.pairs = [(i, j) for i in range(n_generators) for j in range(n_discriminators)]
        self.generators = list(range(n_generators))
        self.discriminators = list(range(n_discriminators))

    def __iter__(self):
        return self

    def __next__(self):
        if self.sampling == 'all_alternated':
            upd = set()
            use = set()
            pairs = []
            for G in self.generators:
                for D in self.discriminators:
                    pairs.append((G, D))
                    use.add(('G', G))
                    use.add(('D', D))
                    if self.n_computations % 2 == 0:
                        upd.add(('D', D))
                    else:
                        upd.add(('G', G))
            self.n_computations += 1
            return pairs, use, upd, False
        if self.sampling == 'all':
            upd = set()
            use = set()
            pairs = []
            for G in self.generators:
                for D in self.discriminators:
                    pairs.append((G, D))
                    use.add(('G', G))
                    use.add(('D', D))
                    if self.n_computations % 4 in [0, 3]:
                        upd.add(('D', D))
                    else:
                        upd.add(('G', G))
            self.n_computations += 1
            return pairs, use, upd, self.n_computations % 2 == 0
        elif self.sampling == 'pair':
            if self.shuffle:
                if self.i == len(self.pairs):
                    self.random_state.shuffle(self.pairs)
                    self.i = 0
            G, D = self.pairs[self.i]
            self.n_computations += 1
            if (self.n_computations - 1) % 4 == 0:
                return [(G, D)], {('D', D), ('G', G)}, {('D', D), }, True
            elif (self.n_computations - 1) % 4 == 1:
                return [(G, D)], {('D', D), ('G', G)}, {('G', G), }, False
            elif (self.n_computations - 1) % 4 == 2:
                return [(G, D)], {('D', D), ('G', G)}, {('G', G), }, True
            else:
                self.i += 1
                return [(G, D)], {('D', D), ('G', G)}, {('D', D), }, False
        elif self.sampling == 'player':
            if self.shuffle:
                if self.i == len(self.generators):
                    self.random_state.shuffle(self.generators)
                    self.i = 0
                if self.j == len(self.discriminators):
                    self.random_state.shuffle(self.discriminators)
                    self.j = 0
            self.n_computations += 1
            if (self.n_computations - 1) % 4 == 0:
                G = self.generators[self.i]
                use = {('G', G), }
                upd = set()
                pairs = []
                for D in self.discriminators:
                    pairs.append((G, D))
                    use.add(('D', D))
                    upd.add(('D', D))
                return pairs, use, upd, True
            elif (self.n_computations - 1) % 4 == 1:
                G = self.generators[self.i]
                use = {('G', G), }
                upd = {('G', G), }
                pairs = []
                for D in self.discriminators:
                    pairs.append((G, D))
                    use.add(('D', D))
                self.i += 1
                return pairs, use, upd, False
            elif (self.n_computations - 1) % 4 == 2:
                D = self.discriminators[self.j]
                use = {('D', D), }
                upd = set()
                pairs = []
                for G in self.generators:
                    pairs.append((G, D))
                    use.add(('G', G))
                    upd.add(('G', G))
                return pairs, use, upd, True
            else:
                D = self.discriminators[self.j]
                use = {('D', D), }
                upd = {('D', D), }
                pairs = []
                for G in self.generators:
                    pairs.append((G, D))
                    use.add(('G', G))
                self.j += 1
                return pairs, use, upd, False
        else:
            raise ValueError()


def enable_grad_for(players, enabled_players=None):
    for group, these_players in players.items():
        for P, player in these_players.items():
            requires_grad = enabled_players is None or (group, P) in enabled_players
            for p in player.parameters():
                p.requires_grad = requires_grad
