from collections import defaultdict
import glob
import math
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


class BurnamentData(object):
    def __init__(self):
        self._project_dir = os.path.expanduser('~/discord_bots')
        self._arc69_dir = os.path.join(self._project_dir, 'arc69')
        self._non_trait_cols = [
            'name', 'asa', 'rank', 'rarity_score', 'unit_name'
        ]
        self._arc69_df = pd.read_csv(
            os.path.join(self._arc69_dir, 'aga_burnament_arc69.csv'),
            header=0,
            index_col=None
        )

        self._trait_cols = [
            col for col in self.arc69_df.columns
            if col not in self.non_trait_cols
        ]
        self._holder_dir = os.path.join(self.project_dir,'holder_data')
        self._aga_holders_df = pd.read_csv(
            os.path.join(self._holder_dir, 'aga_holders.csv'),
            header=0,
            index_col=None
        )
        self._cache_dir = os.path.join(
            self._project_dir,
            'algorillas',
            'bot_cache'
        )
        self._cache = os.path.join(
            self._cache_dir,
            'registered_users.csv'
        )
        if os.path.exists(self._cache):
            self._cache_df = pd.read_csv(
                self._cache,
                header=0,
                index_col=None
            )
        else:
            self._cache_df = None
        self.trait_rarities = {}
        self._round_winners = None
        self._round_matchups = None

    @property
    def aga_holder_df(self):
        """List of AGA holders"""
        return self._aga_holders_df

    @aga_holder_df.setter
    def aga_holder_df(self, value):
        self._aga_holders_df = value

    @property
    def arc69_dir(self):
        """Directory with arc69 data"""
        return self._arc69_dir

    @arc69_dir.setter
    def arc69_dir(self, value):
        self._arc69_dir = value

    @property
    def arc69_df(self):
        """arc69 DataFrame"""
        return self._arc69_df

    @arc69_df.setter
    def arc69_df(self, value):
        self._arc69_df = value

    @property
    def cache(self):
        """Cache directory"""
        return self._cache

    @cache.setter
    def cache(self, value):
        self._cache = value

    @property
    def cache_df(self):
        """Cache DataFrame"""
        return self._cache_df

    @cache_df.setter
    def cache_df(self, value):
        self._cache_df = value

    @property
    def cache_dir(self):
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value):
        self._cache_dir = value

    @property
    def holder_dir(self):
        """Directory with holder data"""
        return self._holder_dir

    @holder_dir.setter
    def holder_dir(self, value):
        self._holder_dir = value

    @property
    def non_trait_cols(self):
        return self._non_trait_cols

    @non_trait_cols.setter
    def non_trait_cols(self, value):
        self._non_trait_cols = value

    @property
    def project_dir(self):
        """Discord bot project directory"""
        return self._project_dir

    @project_dir.setter
    def project_dir(self, value):
        self._project_dir = value

    @property
    def round_winners(self):
        return self._round_winners

    @round_winners.setter
    def round_winners(self, value):
        self._round_winners = value

    @property
    def trait_cols(self):
        return self._trait_cols

    @trait_cols.setter
    def trait_cols(self, value):
        self._trait_cols = value

    @property
    def trait_rarities(self):
        return self._trait_rarities

    @trait_rarities.setter
    def trait_rarities(self, value):
        self._trait_rarities = value

    def compute_trait_rarities(self):
        for trait in self.trait_cols:
            self.trait_rarities[trait] = {}
            for grp, df in self.arc69_df.groupby(trait):
                self.trait_rarities[trait][grp] = df.shape[0] / self.arc69_df.shape[0]
        rarity_score = []
        for i, row in self.arc69_df.iterrows():
            score = 0
            row = row.dropna()
            for trait in self.trait_cols:
                if trait in row.index:
                    score += 1 / self.trait_rarities[trait][row[trait]]
            rarity_score.append(score)

        self.arc69_df['rarity_score'] = rarity_score
        self.arc69_df.sort_values(by='rarity_score', inplace=True, ascending=False)
        self.arc69_df['rank'] = [i + 1 for i in range(self.arc69_df.shape[0])]

    def add_user(self, user_id, wallet):
        if self.cache_df is None:
            tmp = {
                'user_id': [user_id],
                'wallet': [wallet],
                'aga': [None]
            }
            self.cache_df = pd.DataFrame(tmp)

    def load_competitors(self, N):
        flist = glob.glob(f"{self.cache_dir}/*txt")
        competitors = defaultdict(list)
        for f in tqdm(flist[:N]):
            with open(f, 'r') as fobj:
                data = fobj.readlines()[0].split(',')
                competitors['user'].append(data[0])
                competitors['unit_name'].append(data[1])
                competitors['wallet'].append(data[2])
        self.cache_df = pd.DataFrame(competitors)
        # entrants = self.arc69_df.to_dict(orient='list')
        final_list = defaultdict(list)
        for i, row in self.cache_df.iterrows():
            aga_cut = self.arc69_df[self.arc69_df.unit_name == row['unit_name']]
            for col in aga_cut.columns:
                final_list[col].append(aga_cut[col].iloc[0])
            final_list['user'].append(row['user'])
            final_list['wallet'].append(row['wallet'])

        self.entrants = pd.DataFrame(final_list)
        self.entrants = self.entrants.sort_values(by='rank', ascending=True)
        seed = [i+1 for i in range(self.entrants.shape[0])]
        self.entrants['seed'] = seed

    def find_holder(self, aga):
        holder = self.aga_holder_df[self.aga_holder_df.unit_name == aga]
        return holder


    def initialize_bracket(self):
        """Initialize the bracket rounds"""
        self.entrants = self.entrants.sort_values('rank')
        N = self.entrants.shape[0]
        N_desired = 2**(np.ceil(np.log2(N)))

    #     if N > N_desired//2:
        N_byes = int(N_desired - N)
        print(N_byes)
        if N_byes != 0:
            cols = self.entrants.columns
            data = [None]*len(cols)


    #         lowest_seed = entrants['seed'].min()
            j = 0
            for i in range(N, int(N_desired)):
                data[0] = f'BYE{j+1:0.0f}'
                data[-4] = f'BYE{j+1:0.0f}'
                data[-1] = i+1
                data[-6] = 0
                self.entrants.loc[len(self.entrants.index)] = data
                j+=1
        N = self.entrants.shape[0]
        groups = self.generate_tournament(N)
        for key in groups.keys():
            for i, pair in enumerate(groups[key]):
                aga1 = self.entrants[self.entrants['seed'] == pair[0]].iloc[0]
                aga2 = self.entrants[self.entrants['seed'] == pair[1]].iloc[0]
                groups[key][i] = (aga1, aga2)

        self.matchups = groups


    def tournament_round(self, no_of_teams, matchlist):
        new_matches = []
        for team_or_match in matchlist:
            if type(team_or_match) == type([]):
                new_matches += [self.tournament_round(no_of_teams, team_or_match)]
            else:
                new_matches += [
                    [team_or_match, no_of_teams + 1 - team_or_match]]
        return new_matches

    def flatten_list(self, matches):
        teamlist = []
        for team_or_match in matches:
            if type(team_or_match) == type([]):
                teamlist += self.flatten_list(team_or_match)
            else:
                teamlist += [team_or_match]
        return teamlist

    def generate_tournament(self,num):
        num_rounds = math.log(num, 2)
        if num_rounds != math.trunc(num_rounds):
            raise ValueError("Number of teams must be a power of 2")
        teams = 1
        result = [1]
        while teams != num:
            teams *= 2
            result = self.tournament_round(teams, result)
        result = self.flatten_list(result)
        pairs = list(zip(result[::2], result[1::2]))
        groups = defaultdict(list)
        k = 0
        group_size = num // 4
        for i, p in enumerate(pairs):
            if i % group_size == 0 and i != 0:
                k += 1
            groups[k].append(p)
        return groups