from collections import Counter, defaultdict
import os

import discord
from discord.ext import commands
import numpy as np
import pandas as pd


from burnament_helper import BurnamentData



_BOT_TOKEN = os.environ['ALGORILLAS']

_BOOSTED_TRAITS = {
    'Claymore':0.15,
    'Cutlass': 0.2,
    'Spear':0.15,
    'Green Lasers':0.15,
    'Blue Lasers':0.15,
    'Yellow Lasers':0.15
}
bot = commands.Bot(command_prefix='$')

# initialize the burnament data class
_BURN_DATA = BurnamentData()
_BURN_DATA.compute_trait_rarities()

# _BURN_DATA.arc69_df = pd.read_csv(
#     os.path.expanduser('~/algorand_nfts/arc69_data/aga_burnament_arc69.csv'),
#     header=0,
#     index_col=None
# )


_ROUND_WINNERS = defaultdict(list)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

async def register_help(ctx):
    msg = 'Algorillas Burnament Bot Help: **$register**\n'
    msg += ("This bot is used to register a wallet address and AGA with a" 
           " discord user. You will only be able to register AGA that"
            " you own. ")
    await ctx.send(msg)
    msg +=('Usage:\n' \
          '**$register** <AGA> <wallet_address>')
    await ctx.send(msg)

@bot.command(name='register')
async def register(ctx, *args):
    if args[0] == 'help':
        await register_help(ctx)
        return
    if len(args) !=2:
        await ctx.send('Incorret number of arguments. Use the command'
                       '$register help to see more info.')
        return

    user_id = ctx.author
    aga = args[0]
    if 'AGA' not in aga:
        aga = f'AGA{aga}'

    aga_cut = _BURN_DATA.arc69_df[_BURN_DATA.arc69_df.unit_name==aga]
    if aga_cut.empty:
        await ctx.send(f'User input does not correspond to AGA NFT.')
        return

    wallet = args[1]
    wallet_str = f'{wallet[:4]}...{wallet[-4:]}'

    # check what NFTs this wallet holds to make sure they can register this AGA
    holder = _BURN_DATA.aga_holder_df.groupby('address').get_group(wallet)
    if aga in holder['unit_name'].tolist():
        await ctx.send(f'{wallet_str} owns {aga} \u2705')
    else:
        await ctx.send(f'{wallet_str} does not own {aga} \N{cross mark}')
        await ctx.send('You can only register an AGA that you own')
        return

    user_file = f'{_BURN_DATA.cache_dir}/{wallet}.txt'
    if os.path.exists(user_file):
        await ctx.send(f'Updating registration for {user_id}')

    with open(f'{_BURN_DATA.cache_dir}/{wallet}.txt', 'w+') as fobj:
        fobj.write(f'{user_id},{aga},{wallet}')

    await ctx.send('Registration successful')
    # await ctx.send(f'Hello {ctx.author}!')

@bot.command(name='aga_info')
async def aga_info(ctx, aga):
    if 'AGA' not in aga:
        aga = f"AGA{aga}"
    aga = _BURN_DATA.arc69_df[_BURN_DATA.arc69_df.unit_name == aga].iloc[0]
    nontrait_cols = ['unit_name','asa', 'rank', 'rarity_score', ]
    trait_cols = [
        col for col in _BURN_DATA.arc69_df.columns if col not in nontrait_cols
    ]
    msg = ""
    for col, l in zip(nontrait_cols, ['Name','ASA', 'Rank','Rarity Score']):
        if l == 'Rarity Score':
            msg += f"**{l}**: {aga[col]:0.3f}\n"
        else:
            msg += f"**{l}**: {aga[col]}\n"

    await ctx.send(msg)
    msg = f"{'**Traits**'.center(50,'-')}\n"
    for col in trait_cols:
        msg += f"**{col.capitalize()}**: {aga[col]}\n"

    await ctx.send(msg)

@bot.command(name='load_burnament')
async def load_burnament(ctx, *args):
    _BURN_DATA.load_competitors(int(args[0]))

    _BURN_DATA.initialize_bracket()
    msg = f"Burnament Loaded!\n"
    msg +=f"{'**Top 10 Seeds**'.center(50,'-')}\n"
    j = 0
    for i, row in _BURN_DATA.entrants.iterrows():
        if j == 10 or j > _BURN_DATA.entrants.shape[0]:
            break
        msg += (f"({row['seed']}) {row['name']}, "
                f"**Rank:** {row['rank']}, **Holder:** {row['user']}\n")
        j+=1
    await ctx.send(msg)

# @bot.command(name='fight')
# async def fight(ctx, *args):
#     print(_BURN_DATA.matchups.keys())
#     top_half = _BURN_DATA.matchups['Round of 16']['top']
#     picks = top_half
#     N_rounds = np.log2(len(picks) * 2) - 1
#     winners1 = defaultdict(list)
#     winners2 = defaultdict(list)
#     i = 0
#     while i < N_rounds:
#         if i == 0:
#             winners = []
#             for m in picks:
#                 w = await pick_winner(
#                     ctx,
#                     m[0]['unit_name'],
#                     m[1]['unit_name']
#                 )
#                 winners.append(w[-1])
#             # history_of_winners.append(winners)
#             winners1[i] += [
#                 _BURN_DATA.entrants[_BURN_DATA.entrants['unit_name'] == w].iloc[0]
#                 for w in winners
#             ][::2]
#             winners2[i] += [
#                 _BURN_DATA.entrants[_BURN_DATA.entrants['unit_name'] == w].iloc[0]
#                 for w in winners
#             ][1::2]
#             i += 1
#             continue
#         winners = []
#         for k in range(len(winners1[i - 1]) // 2):
#             cut = slice(k, k + 2)
#             m = winners1[i - 1][cut]
#             w = await pick_winner(
#                 ctx,
#                 m[0]['unit_name'],
#                 m[1]['unit_name']
#             )
#             winners.append(w[-1])
#         winners1[i] += [
#                 _BURN_DATA.entrants[_BURN_DATA.entrants['unit_name'] == w].iloc[0]
#                 for w in winners
#         ][::2]
#
#         winners = []
#         for k in range(len(winners2[i - 1]) // 2):
#             cut = slice(k, k + 2)
#             m = winners2[i - 1][cut]
#             w = await pick_winner(
#                 ctx,
#                 m[0]['unit_name'],
#                 m[1]['unit_name']
#             )
#             winners.append(w[-1])
#         winners2[i] += [
#             _BURN_DATA.entrants[_BURN_DATA.entrants['unit_name'] == w].iloc[0]
#             for w in winners
#         ][::2]
#         i+=1

@bot.command(name='run_round')
async def run_round(ctx, args):
    round_name = args[0]
    if _BURN_DATA.round_winners is None:
        _BURN_DATA.round_winners = {'top': {}, 'bottom': {}}
        _BURN_DATA.round_matchups = {'top': {}, 'bottom': {}}

    for k in [0,1]:
        matches = _BURN_DATA.matchups[k]
        round_num = 0
        # while n_competitors >=2:
        round_winners = []
        for m in matches:
            w = await pick_winner(
                ctx,
                m[0]['unit_name'],
                m[1]['unit_name'],
                verbose=True
            )
            round_winners.append(w[-1])
        round_winners = [
            _BURN_DATA.entrants[_BURN_DATA.entrants.unit_name == w].iloc[0]
            for w in round_winners
        ]
        matches = list(zip(round_winners[::2], round_winners[1::2]))
        _BURN_DATA.round_winners[round_name] = round_winners
        round_num += 1
        n_competitors = 2 * len(matches)
    # print('-' * 100)

@bot.command(name='fight')
async def fight(ctx, *args):

    # n_competitors = 2*len(_BURN_DATA.matchups[0])
    # matches = _BURN_DATA.matchups[0]
    round_num = 0
    top_half_winner = None
    bottom_half_winner = None
    winners = defaultdict(dict)
    for k in [0,1]:
        if k == 0:
            await ctx.send('**Running top half of the draw**')
        else:
            await ctx.send('**Running bottom half of the draw**')
        winners[k]
        n_competitors = 2 * len(_BURN_DATA.matchups[k])
        matches = _BURN_DATA.matchups[k]
        round_num = 0
        while n_competitors >=2:
            round_winners = []
            for m in matches:
                w = await pick_winner(
                    ctx,
                    m[0]['unit_name'],
                    m[1]['unit_name'],
                    verbose=True
                )
                round_winners.append(w[-1])
            # msg = 'ROUND WINNERS:\n'+'\n'.join(round_winners)
            # await ctx.send(msg)
            round_winners = [
                _BURN_DATA.entrants[_BURN_DATA.entrants.unit_name == w].iloc[0]
                for w in round_winners
            ]

            matches = list(zip(round_winners[::2], round_winners[1::2]))
            winners[k][round_num] = round_winners
            round_num +=1
            n_competitors = 2*len(matches)

    top_half_winner = winners[0][round_num-1][0]
    bottom_half_winner = winners[1][round_num-1][0]
    #
    await ctx.send(
        (
            '**Final two standing**\n '
            f'({top_half_winner["seed"]}){top_half_winner["name"]} \n'
            f'({bottom_half_winner["seed"]}) {bottom_half_winner["name"]}'
         )
    )
    w = await pick_winner(
        ctx,
        top_half_winner['unit_name'],
        bottom_half_winner['unit_name']
    )
    winner = w[-1]
    await ctx.send(f'**{winner} is the champion**'.center(30, '\N{PARTY POPPER}'))

@bot.command(name='battle')
async def pick_winner(ctx, aga1, aga2, verbose=True):

    await ctx.send("-" * 40)
    if 'AGA' not in aga1 and 'BYE' not in aga1:
        aga1 = f'AGA{aga1}'

    if 'AGA' not in aga2 and 'BYE' not in aga2:
        aga2 = f'AGA{aga2}'

    aga1 = _BURN_DATA.entrants[_BURN_DATA.entrants.unit_name == aga1].iloc[0]
    aga2 = _BURN_DATA.entrants[_BURN_DATA.entrants.unit_name == aga2].iloc[0]
    s1_rs = aga1['rarity_score']
    s1_name = aga1['name']
    s1_unit_name = aga1['unit_name']
    try:
        s1_seed = aga1['seed']
    except KeyError:
        s1_seed = ''
    s1_item = aga1['item']
    s1_rank = aga1['rank']
    s1_accessory = aga1['accessories']
    s1_boost = 0
    if s1_item in _BOOSTED_TRAITS.keys():
        if verbose:
            await ctx.send(f"{s1_item} is a boosted trait!")
            await ctx.send(f"Modifying rarity score for {s1_name}")
        s1_boost += _BOOSTED_TRAITS[s1_item]

    if s1_accessory in _BOOSTED_TRAITS.keys():
        if verbose:
            await ctx.send(f"{s1_accessory} is a boosted trait!")
            await ctx.send(f"Modifying rarity score for {s1_name}")
        s1_boost += _BOOSTED_TRAITS[s1_accessory]

    s2_rs = aga2['rarity_score']
    s2_name = aga2['name']
    s2_unit_name = aga2['unit_name']
    try:
        s2_seed = aga2['seed']
    except KeyError:
        s2_seed = ''
    s2_item = aga2['item']
    s2_rank = aga2['rank']
    s2_accessory = aga2['accessories']
    s2_boost = 0
    if s2_item in _BOOSTED_TRAITS.keys():
        if verbose:
            await ctx.send(f"{s2_item} is a boosted trait!")
            await ctx.send(f"Modifying rarity score for {s2_name}")
        s2_boost += _BOOSTED_TRAITS[s2_item]

    if s2_accessory in _BOOSTED_TRAITS.keys():
        if verbose:
            await ctx.send(f"{s2_accessory} is a boosted trait!")
            await ctx.send(f"Modifying rarity score for {s2_name}")
        s2_boost += _BOOSTED_TRAITS[s2_accessory]

    s1_score = np.round(s1_rs * (1 + s1_boost), 0)
    s2_score = np.round(s2_rs * (1 + s2_boost), 0)
    p1_odds = s1_score / (s1_score + s2_score)
    p2_odds = s2_score / (s1_score + s2_score)
    if s1_seed and s2_seed:
        s1_seed_msg = f'({s1_seed})'
        s2_seed_msg = f'({s2_seed})'
    else:
        s1_seed_msg = ''
        s2_seed_msg = ''

    msg = (
        f"{s1_seed_msg} {s1_name} with rank {s1_rank} has {p1_odds:.2%} of winning\n"
        f"{s2_seed_msg} {s2_name} with rank {s2_rank} has {p2_odds:.2%} of winning\n"
    )
    if verbose:
        await ctx.send(msg)
    entries = [s1_name] * int(s1_score) + [s2_name] * int(s2_score)
    results = np.random.choice(entries, 5)
    c = Counter(results)
    winner = c.most_common(1)[0][0]
    a = '\n'.join(results)
    if winner == s1_name:
        winner_seed = s1_seed_msg
        w = s1_unit_name
    else:
        winner_seed = s2_seed_msg
        w = s2_unit_name

    if verbose:
        await ctx.send(f"Draws\n {a}")
        await ctx.send("-" * 40)
        await ctx.send(f"Winner: {winner_seed} {winner}\n")

    return results, entries, w

@bot.command(name='simulate_burnament')
async def simulate_burnament(ctx):
    round_winners = {}
    entrants = _BURN_DATA.arc69_df.sample(512)
    contestants = entrants['unit_name'].values
    # round_names = {5:'Quarterfinals', 6:'Semifinals', 7:'Finals', 8:'Champion'}
    for round_num in range(1, 10):
        winners = []
        if round_num == 1:
            entrants = contestants
        else:
            entrants = round_winners[round_num - 1]
        start_idx = 0
        stop_idx = start_idx + 2
        for i in range(len(entrants) // 2):
            cut = slice(start_idx, stop_idx)
            opponents = entrants[cut]
            # aga1 = aga_arc69['unit_name'] == opponents[0]
            # aga2 = aga_arc69[aga_arc69['unit_name'] == opponents[1]]
            results, entries, winner = await pick_winner(
                ctx,
                opponents[0],
                opponents[1],
                verbose=False
            )
            winners.append(winner)
            start_idx = stop_idx
            stop_idx = start_idx + 2
        if len(winners) < 16:
            if len(winners) == 8:
                round_name = 'Quarterfinals'
            elif len(winners) == 4:
                round_name = 'Semifinals'
            elif len(winners) == 2:
                round_name = 'Finals'
            elif len(winners) == 1:
                round_name = 'Champion'
            winners_str = '\n'.join(winners)
            await ctx.send(f'{round_name}:\n{winners_str}')
            await ctx.send('-' * 30)
        round_winners[round_num] = winners


bot.run(_BOT_TOKEN)