from sklearn.utils import check_random_state


def Scheduler(n_generators, n_discriminators, random_state=None, shuffle=True, sampling='pair'):
    random_state = check_random_state(random_state)
    n_computations = 0
    i, j = 0, 0
    if sampling == 'all_alternated':
        while True:
            generators = list(range(n_generators))
            discriminators = list(range(n_discriminators))
            upd = set()
            use = set()
            pairs = []
            for G in generators:
                for D in discriminators:
                    pairs.append((G, D))
                    use.add(('G', G))
                    use.add(('D', D))
                    if n_computations % 2 == 0:
                        upd.add(('D', D))
                    else:
                        upd.add(('G', G))
            yield pairs, use, upd, False
            n_computations += 1
    if sampling == 'all':
        while True:
            generators = list(range(n_generators))
            discriminators = list(range(n_discriminators))
            upd = set()
            use = set()
            pairs = []
            for G in generators:
                for D in discriminators:
                    pairs.append((G, D))
                    use.add(('G', G))
                    use.add(('D', D))
                    if n_computations % 4 in [0, 3]:
                        upd.add(('D', D))
                    else:
                        upd.add(('G', G))
            yield pairs, use, upd, n_computations % 2 == 0
            n_computations += 1
    elif sampling == 'pair':
        pairs = [(i, j) for i in range(n_generators) for j in range(n_discriminators)]
        while True:
            if shuffle:
                if i == n_generators * n_discriminators:
                    random_state.shuffle(pairs)
                    i = 0
            G, D = pairs[i]
            yield [(G, D)], {('D', D), ('G', G)}, {('D', D), }, True
            yield [(G, D)], {('D', D), ('G', G)}, {('G', G), }, False
            yield [(G, D)], {('D', D), ('G', G)}, {('G', G), }, True
            yield [(G, D)], {('D', D), ('G', G)}, {('D', D), }, False
            n_computations += 4
            i += 1
    elif sampling == 'player':
        generators = list(range(n_generators))
        discriminators = list(range(n_discriminators))
        while True:
            if shuffle:
                if i == n_generators:
                    random_state.shuffle(generators)
                    i = 0
                if j == n_discriminators:
                    random_state.shuffle(discriminators)
                    j = 0
            if n_computations % 2 == 0:
                G = generators[i]
                use = {('G', G), }
                upd = set()
                pairs = []
                for D in discriminators:
                    pairs.append((G, D))
                    use.add(('D', D))
                    upd.add(('D', D))
                yield pairs, use, upd, True
                upd = {('G', G), }
                use = {('G', G), }
                pairs = []
                for D in discriminators:
                    pairs.append((G, D))
                    use.add(('D', D))
                yield pairs, use, upd, False
                i += 1
                n_computations += 1
            else:
                D = discriminators[j]
                use = {('D', D), }
                upd = set()
                pairs = []
                for G in generators:
                    pairs.append((G, D))
                    use.add(('G', G))
                    upd.add(('G', G))
                yield pairs, use, upd, True
                upd = {('D', D), }
                use = {('D', D), }
                pairs = []
                for G in generators:
                    pairs.append((G, D))
                    use.add(('G', G))
                yield pairs, use, upd, False
                j += 1
                n_computations += 1
    else:
        raise ValueError()


def enable_grad_for(players, enabled_players=None):
    for group, these_players in players.items():
        for P, player in these_players.items():
            requires_grad = enabled_players is None or (group, P) in enabled_players
            for p in player.parameters():
                p.requires_grad = requires_grad
