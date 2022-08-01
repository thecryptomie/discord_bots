
from collections import defaultdict
import glob
import logging
import math
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

# Set up the style for logging output
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s')

# instantiate the logger
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)


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
        self._holder_dir = os.path.join(self.project_dir, 'holder_data')
        self._aga_holders_df = pd.read_csv(
            os.path.join(self._holder_dir, 'aga_holders.csv'),
            header=0,
            index_col=None
        )
        self._aga_holders_df = pd.merge(
            self._aga_holders_df,
            self._arc69_df,
            on='asa',
            how='inner',
            suffixes=('', '_1')
        )
        self._aga_holders_df= self._aga_holders_df.drop(
            columns=['name_1', 'unit_name_1']
        )
        t1 = self._aga_holders_df.groupby('address').get_group(
            'PO4CEJB6IV2P5UACZ3P77KJCITMX2ZIT6RMW4WTX6JQGJYNJS6T5E4V27Q'
        )
        t1['creator_address'] = [
            'PO4CEJB6IV2P5UACZ3P77KJCITMX2ZIT6RMW4WTX6JQGJYNJS6T5E4V27Q'
        ] * t1.shape[0]

        t2 = self._aga_holders_df.groupby('address').get_group(
            'MPRRGD2IXHYNHRMOFD5AE6Y2KK6DL32GKDFIZG7SC6TYO6AKK7CZSSBKTA'
        )
        t2['creator_address'] = [
            'MPRRGD2IXHYNHRMOFD5AE6Y2KK6DL32GKDFIZG7SC6TYO6AKK7CZSSBKTA'
        ] * t2.shape[0]

        self._unreleased_aga_df = pd.concat([t1, t2], axis=0)
        self._unreleased_aga_df = self._unreleased_aga_df.sample(frac=1)
        self._cache_dir = os.path.join(
            self._project_dir,
            'algorillas',
            'bot_cache'
        )
        self._entrants_dir = os.path.join(
            self._cache_dir,
            'entrants'
        )
        self._giveaway_dir = os.path.join(
            self._cache_dir,
            'giveaway_winners'
        )
        self._tournament_results_dir = os.path.join(
            self._cache_dir,
            'tournament_results'
        )
        self._cache = os.path.join(
            self._entrants_dir,
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
        self._matchups = None
        self._round_winners = None
        self._round_history = None
        self._round_names = []

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
    def entrants_dir(self):
        return self._entrants_dir

    @entrants_dir.setter
    def entrants_dir(self, value):
        self._entrants_dir

    @property
    def cache_dir(self):
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value):
        self._cache_dir = value
    @property
    def giveaway_dir(self):
        return self._giveaway_dir

    @giveaway_dir.setter
    def giveaway_dir(self, value):
        self._giveaway_dir = value

    @property
    def holder_dir(self):
        """Directory with holder data"""
        return self._holder_dir

    @holder_dir.setter
    def holder_dir(self, value):
        self._holder_dir = value
    @property
    def matchups(self):
        return self._matchups

    @matchups.setter
    def matchups(self, value):
        self._matchups = value

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
    def round_history(self):
        return self._round_history

    @round_history.setter
    def round_history(self, value):
        self._round_history = value



    @property
    def round_names(self):
        return self._round_names

    @round_names.setter
    def round_names(self, value):
        self._round_names = value

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
    def tournament_results_dir(self):
        return self._tournament_results_dir

    @tournament_results_dir.setter
    def tournament_results_dir(self, value):
        self._tournament_results_dir

    @property
    def trait_rarities(self):
        return self._trait_rarities

    @trait_rarities.setter
    def trait_rarities(self, value):
        self._trait_rarities = value

    @property
    def unreleased_aga_df(self):
        return self._unreleased_aga_df

    @unreleased_aga_df.setter
    def unreleased_aga_df(self, value):
        self._unreleased_aga_df = value

    def compute_trait_rarities(self):
        for trait in self.trait_cols:
            self.trait_rarities[trait] = {}
            for grp, df in self.arc69_df.groupby(trait):
                self.trait_rarities[trait][grp] = df.shape[0] / \
                                                  self.arc69_df.shape[0]
        rarity_score = []
        for i, row in self.arc69_df.iterrows():
            score = 0
            row = row.dropna()
            for trait in self.trait_cols:
                if trait in row.index:
                    score += 1 / self.trait_rarities[trait][row[trait]]
            rarity_score.append(score)

        self.arc69_df['rarity_score'] = rarity_score
        self.arc69_df.sort_values(by='rarity_score', inplace=True,
                                  ascending=False)
        self.arc69_df['rank'] = [i + 1 for i in range(self.arc69_df.shape[0])]

    def add_user(self, user_id, wallet):
        if self.cache_df is None:
            tmp = {
                'user_id': [user_id],
                'wallet': [wallet],
                'aga': [None]
            }
            self.cache_df = pd.DataFrame(tmp)

    def load_users(self):
        """Load the registered users

        Returns
        -------

        """
        flist = glob.glob(self.entrants_dir + '/*txt')
        users = defaultdict(list)
        for f in tqdm(flist):
            with open(f, 'r') as fobj:
                data = fobj.readlines()[0].split(',')
                users['user'].append(data[0])
                users['wallet'].append(data[1])
        return users

    def load_competitors(self, N=None):
        flist = glob.glob(f"{self.entrants_dir}/*txt")
        np.random.shuffle(flist)
        if N is not None:
            flist = flist[:N]
        competitors = defaultdict(list)
        for f in tqdm(flist):
            with open(f, 'r') as fobj:
                data = fobj.readlines()[0].split(',')
                agas = data[2:]
                for aga in agas:
                    competitors['user'].append(data[0])
                    competitors['wallet'].append(data[1])
                    competitors['unit_name'].append(aga)

        self.cache_df = pd.DataFrame(competitors)
        # entrants = self.arc69_df.to_dict(orient='list')
        final_list = defaultdict(list)
        for i, row in self.cache_df.iterrows():
            aga_cut = self.arc69_df[
                self.arc69_df.unit_name == row['unit_name']]
            for col in aga_cut.columns:
                final_list[col].append(aga_cut[col].iloc[0])
            final_list['user'].append(row['user'])
            final_list['wallet'].append(row['wallet'])

        self.entrants = pd.DataFrame(final_list)
        self.entrants = self.entrants.sort_values(by='rank', ascending=True)
        seed = [i + 1 for i in range(self.entrants.shape[0])]
        self.entrants['seed'] = seed

    def find_holder(self, aga):
        holder = self.aga_holder_df[self.aga_holder_df.unit_name == aga]
        return holder

    def initialize_bracket(self):
        """Initialize the bracket rounds"""
        self.entrants = self.entrants.sort_values('rank')
        N = self.entrants.shape[0]
        N_desired = 2 ** (np.ceil(np.log2(N)))
        N_players = N_desired
        N_players_per_round = []
        while N_players > 1:
            N_players_per_round.append(int(N_players))
            N_players /= 2

        round_names = []
        special_rounds = {8: 'Quarterfinal Round', 4: 'Semifinal Round',
                          2: 'Championship Round'}
        for n in N_players_per_round:
            if n not in special_rounds.keys():
                round_names.append(f'Round of {n}')
            else:
                round_names.append(special_rounds[n])
        #     if N > N_desired//2:
        self.round_names = round_names

        N_byes = int(N_desired - N)

        if N_byes != 0:
            cols = self.entrants.columns
            data = [None] * len(cols)

            #         lowest_seed = entrants['seed'].min()
            j = 0
            for i in range(N, int(N_desired)):
                data[0] = f'BYE{j + 1:0.0f}'
                data[-4] = f'BYE{j + 1:0.0f}'
                data[-1] = i + 1
                data[-6] = 0
                self.entrants.loc[len(self.entrants.index)] = data
                j += 1
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
                new_matches += [
                    self.tournament_round(no_of_teams, team_or_match)]
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

    def generate_tournament(self, num):
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

    def get_round_winners(self, round_name):
        if round_name in self.round_history.keys():
            winners1 = self.round_history[round_name]['top']['winners']
            winners2 = self.round_history[round_name]['bottom']['winners']
            winner_list = [val for val in winners1]
            winner_list += [val for val in winners2]

            return winner_list


    def save_round(self, round_name):
        round_file = (
            f"{self.tournament_results_dir}/{round_name.replace(' ','_')}"
        )
        if round_name == 'Championship Round':
            f = f"{round_file}.pkl"
            with open(f, 'wb+') as fobj:
                pickle.dump(self.round_history[round_name], fobj)

        else:
            for key, item in self.round_history[round_name].items():
                f = f"{round_file}_{key}.pkl"
                LOG.info(f)
                with open(f, 'wb+') as fobj:
                    pickle.dump(item, fobj)

    def load_round(self, round_name):
        if self.round_history is None:
            self.round_history = {
                round_name: {'top': {}, 'bottom': {}}
            }
        round_file = (
            f"{self.tournament_results_dir}/{round_name.replace(' ', '_')}"
        )
        if round_name == 'Championship Round':
            f = f"{round_file}.pkl"
            with open(f, 'rb') as fobj:
                self.round_history[round_name] = pickle.load(fobj)
        else:
            for key in ['top','bottom']:
                f = f"{round_file}_{key}.pkl"
                with open(f, 'rb') as fobj:
                    self.round_history[round_name][key] = pickle.load(fobj)

    async def print_round_matchups(self,round_name, bot):
        msg = ''
        messages = []

        if round_name == 'Championship Round':
            msg +='Championship Matchup'
            matchup = self.matchups["Championship Round"]
            member1 = await bot.fetch_user(matchup[0]['user'])
            member2 = await bot.fetch_user(matchup[1]['user'])
            msg += (
                f"({matchup[0]['seed']}) {matchup[0]['name']}, "
                f"{member1.mention} vs "
                f"({matchup[1]['seed']}) {matchup[1]['name']}, "
                f"{member2.mention}"
            )
            messages.append(msg)
        else:
            top_half = self.matchups[0]
            bottom_half = self.matchups[1]
            # LOG.info(top_half)
            msg += f'**{round_name}**\n'
            title_str = [
                '**Top half of the draw**'.center(50, '-') + '\n',
                '**Bottom half of the draw**'.center(50, '-') + '\n'
            ]
            current_length = 0
            for half, title in zip([top_half, bottom_half], title_str):
                msg += title
                msg_length = 0
                # start_length = sum([len(val) for val in messages])

                for m in half:
                    aga1 = m[0]
                    aga2 = m[1]
                    # LOG.info(aga1)
                    if aga1['user'] is not None:
                        member1 = await bot.fetch_user(m[0]['user'])
                    else:
                        member1 = 'N/A'
                    # LOG.info(aga2)
                    if aga2['user'] is not None:
                        member2 = await bot.fetch_user(m[1]['user'])
                    else:
                        member2 = 'N/A'
                    msg += (
                        f"({aga1['seed']}) {aga1['unit_name']} "
                        " vs "
                        f"({aga2['seed']}) {aga2['unit_name']}\n"
                        f"{member1 if isinstance(member1, str) else member1.mention}"
                        " vs "
                        f"{member2 if isinstance(member2, str) else member2.mention}\n"

                    )
                    msg_length += len(msg)
                    current_length += len(msg)
                    if msg_length > 1000:
                        messages.append(msg)
                        msg = ''
                        msg_length = 0

                msg_list_length = sum([len(val) for val in messages])
                if msg_list_length == 0:
                    messages.append(msg)
                    msg_list_length = sum(
                        [len(val) for val in messages]
                    )
                elif msg_list_length < current_length:
                    messages.append(msg)
                msg = ''
        # print(messages)
        return messages

    async def print_round_summary(self, round_name, bot):
        msg = ''
        messages = []
        if round_name in self.round_history.keys():
            msg += round_name + '\n'
            if round_name == 'Championship Round':
                champ_round = self.round_history[round_name]
                matchup = champ_round['matches']
                winner = champ_round['winners']
                msg += 'Championship Matchup\n'
                msg += (
                    f"({matchup[0]['seed']}) {matchup[0]['unit_name']} vs "
                    f"({matchup[1]['seed']}) {matchup[1]['unit_name']}"
                )
                messages.append(msg)
            else:
                top_half = self.round_history[round_name]['top']
                bottom_half = self.round_history[round_name]['bottom']
                title_str = [
                    '**Top half of the draw**'.center(50, '-') + '\n',
                    '**Bottom half of the draw**'.center(50, '-') + '\n'
                ]
                current_length = 0
                for half, title in zip([top_half, bottom_half], title_str):
                    msg += title
                    msg_length = 0
                    # start_length = sum([len(val) for val in messages])
                    for m in half['matches']:
                        aga1 = m[0]
                        aga2 = m[1]
                        msg += (f"({aga1['seed']}) {aga1['unit_name']} vs "
                                f"({aga2['seed']}) {aga2['unit_name']}\n")
                        msg_length += len(msg)
                        current_length += len(msg)
                        if msg_length > 1000:
                            messages.append(msg)
                            msg = ''
                            msg_length = 0

                    msg_list_length = sum([len(val) for val in messages])
                    if msg_list_length == 0:
                        messages.append(msg)
                        msg_list_length = sum(
                            [len(val) for val in messages]
                        )
                    elif msg_list_length < current_length:
                        messages.append(msg)

                    msg = ''
                    msg += "**Winners**".center(25, '-') + '\n'
                    count = 0
                    msg_length = 0
                    for w in half['winners']:
                        member = await bot.fetch_user(w['user'])
                        msg += (f"({w['seed']}) {w['name']} **{member.mention}**\n")
                        msg_length += len(msg)
                        current_length += len(msg)
                        if msg_length > 1000:
                            messages.append(msg)
                            msg = ''
                            msg_length = 0
                    msg_list_length = sum([len(val) for val in messages])
                    if msg_list_length < current_length:
                        messages.append(msg)
                    # messages.append('-'*40+'\n')
                    msg = ''

        return messages
