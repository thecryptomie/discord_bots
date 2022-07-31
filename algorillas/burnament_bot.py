import asyncio
from collections import Counter, defaultdict
import glob
import logging
import os
import time

import discord
intents = discord.Intents.default()
intents.members = True

client = discord.Client(intents=intents)
from discord.ext import commands
import numpy as np
import requests


from burnament_helper import BurnamentData

# Set up the style for logging output
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s')

# instantiate the logger
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)
_ALGO_PRIZE_AMOUNT = 500

_BOT_TOKEN = os.environ['ALGORILLAS']

_BOOSTED_TRAITS = {
    'Claymore':0.15,
    'Cutlass': 0.2,
    'Spear':0.15,
    'Green Lasers':0.15,
    'Blue Lasers':0.15,
    'Yellow Lasers':0.15
}
bot = commands.Bot(command_prefix='#')

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

async def get_user_data(ctx):
    member = ctx.author
    user_id = member.id
    user_file = f'{_BURN_DATA.entrants_dir}/{user_id}.txt'
    with open(user_file, 'r') as fobj:
        data = fobj.readlines()[0].split(',')
    user_id = data[0]
    wallet = data[1]
    return user_id, wallet, data[2:]

async def register_help(ctx):
    msg = 'Algorillas Burnament Bot Help: **#register**\n'
    msg += ("This bot is used to register a wallet address with a" 
           " discord user. You will only be able to register one wallet address"
            " to your account. ")
    await ctx.author.send(msg)
    msg =''
    msg +=('Usage:\n' \
          '**#register** <wallet_address or NFD>')
    await ctx.author.send(msg)

# @bot.command(name='member_ids')
# @commands.has_role('ADMIN')
# async def get_member_id(ctx):
#     for discord_handle in _QUALIFIED_HOLDERS['discord']:
#         try:
#             member = await commands.MemberConverter().convert(
#                 ctx,
#                 discord_handle
#             )
#             await ctx.author.send(f"{member.name} {member.id}")
#         except Exception as e:
#             member = discord_handle
#          await ctx.author.send(member.id)

@bot.command(name='load_users')
async def load_users(ctx):
    if str(ctx.author) != 'cauchy69.APE#8518':
        await ctx.send(f'{ctx.author} is an unauthorized user')
        return

    user_data = _BURN_DATA.load_users()
    messages = []
    msg = '**Registered Users**\n'
    msg_length = 0
    for user, wallet in zip(user_data['user'], user_data['wallet']):
        msg += f"{user},{wallet}\n"
        msg_length += len(msg)
        if msg_length > 1500:
            messages.append(msg)
            msg = ''
            current_msg_length = msg_length

    msg_list_length = sum([len(val) for val in messages])
    if msg_list_length == 0:
        messages.append(msg)
    elif msg_list_length < msg_length:
        messages.append(msg)
    for msg in messages:
        await ctx.author.send(msg)

async def lookup_nfd(ctx, *args):
    if '.' in args[0]:
        nfd = args[0].split('.')[0]
        nfd = nfd +'.algo'
    else:
        nfd = args[0] + '.algo'

    url = f"https://api.nf.domains/nfd/{nfd}?view=tiny"
    # print(url)
    response = requests.get(url)
    data = response.json()
    if 'owner' in data.keys():
        wallet = data['owner']
    else:
        await ctx.author.send(f'{nfd} could not be resolved.')
        wallet = 'Not found'
    return wallet

@bot.command(name='register')
async def register(ctx, *args):
    # if str(ctx.author) != 'cauchy69.APE#8518':
    #     await ctx.send(f'Holders airdrop registration is closed.')
    #     return

    if len(args) == 0:
        await register_help(ctx)
        return
    elif args[0] == 'help':
        await register_help(ctx)
        return
    elif len(args) > 1:
        await ctx.author.send('Incorret number of arguments. Use the command'
                       '#register help to see more info.')
        return

    wallet = args[0]
    if len(wallet) != 58:
        wallet = await lookup_nfd(ctx, wallet)

    member = ctx.author
    user_id = member.id
    user_file = f'{_BURN_DATA.entrants_dir}/{user_id}.txt'
    if os.path.exists(user_file):
        await ctx.author.send(
            (f'{member.mention} is already registered. '
            'Use #my_entries to check your registration.')
        )
        await ctx.message.add_reaction('\N{WHITE HEAVY CHECK MARK}')
        return
    if wallet == 'Not found':
        return
    else:
        with open(user_file, 'w+') as fobj:
            fobj.write(f'{user_id},{wallet}')
        await ctx.author.send('Registration successful')
        await ctx.message.add_reaction('\N{WHITE HEAVY CHECK MARK}')

async def unregister_help(ctx):
    msg = 'Algorillas Burnament Bot Help: **#unregister**\n'
    msg += "This bot is used to remove your Burnament registration.\n"
    msg += "Usage:\n"
    msg += "**#unregister**"
    await ctx.author.send(msg)

@bot.command(name='unregister')
async def unregister(ctx, *args):

    if len(args) != 0 and args[0] == 'help':
        await unregister_help(ctx)
        return
    elif len(args)>0:
        await ctx.author.send('Invalid usage. Use #unregister help to see more details')
        return

    user_id = str(ctx.author)
    user_file = f'{_BURN_DATA.entrants_dir}/{user_id}.txt'
    if os.path.exists(user_file):
        await ctx.author.send(f'Removing registration for {user_id}...')
        os.remove(user_file)
        await ctx.message.add_reaction('\N{WHITE HEAVY CHECK MARK}')
        return
    else:
        await ctx.author.send('User not found!')


async def add_aga_help(ctx):
    msg = 'Algorillas Burnament Bot Help: **#add_aga**\n'
    msg += (
        "This bot is used to add an AGA to your list of competitors. "
        "Each registered account can submit as many AGA as they like. "
    )
    await ctx.author.send(msg)
    msg = ''
    msg +=(
        'Usage:\n' 
        '**#add_aga** <aga_number>\n'
        '<aga_number> can be either a number (e.g. 100) '
        'or the unitname (e.g AGA100).'
    )
    await ctx.author.send(msg)

@bot.command('add_aga')
async def add_aga(ctx, *args):
    # if str(ctx.author) != 'cauchy69.APE#8518':
    #     await ctx.author.send(f'Command not available yet. Check back when registration opens!')
    #     return
    if args[0] == 'help':
        await add_aga_help(ctx)
        return

    aga = args[0]
    if 'AGA' not in aga:
        aga = f'AGA{aga}'

    aga_cut = _BURN_DATA.arc69_df[_BURN_DATA.arc69_df.unit_name==aga]
    if aga_cut.empty:
        await ctx.author.send(f'User input does not correspond to AGA NFT.')
        return
    member = ctx.author
    user_id = member.id
    user_file = f'{_BURN_DATA.entrants_dir}/{user_id}.txt'
    with open(user_file,'r') as fobj:
        data = fobj.readlines()[0].split(',')
    user_id = data[0]
    wallet = data[1]
    wallet_str = f'{wallet[:4]}...{wallet[-4:]}'

    # check what NFTs this wallet holds to make sure they can register this AGA
    holder = _BURN_DATA.aga_holder_df.groupby('address').get_group(wallet)
    if aga in holder['unit_name'].tolist():
        # await ctx.author.send(f'{wallet_str} owns {aga} \u2705')
        await ctx.message.add_reaction('\N{WHITE HEAVY CHECK MARK}')
    else:
        await ctx.message.add_reaction('\N{CROSS MARK}')
        # await ctx.author.send(f'{wallet_str} does not own {aga} \N{cross mark}')
        # await ctx.author.send('You can only register an AGA that you own')
        return

    if aga in data[2:]:
        await ctx.author.send(f'{aga} already submitted for battle!')
        return
    else:
        await ctx.author.send(f'Adding {aga} to {member.mention} competitor list.')
        with open(user_file,'a+') as fobj:
            fobj.write(f',{aga}')


async def remove_aga_help(ctx):
    msg = 'Algorillas Burnament Bot Help: **#remove_aga**\n'
    msg += (
        "This bot is used to remove an AGA from your list of competitors. "
    )
    await ctx.author.send(msg)
    msg=''
    msg +=(
        'Usage:\n' 
        '**#remove_aga** <aga_number>\n'
        '<aga_number> can be either a number (e.g. 100) '
        'or the unitname (e.g AGA100).'
    )
    await ctx.author.send(msg)

@bot.command('remove_aga')
async def remove_aga(ctx, *args):
    if args[0] == 'help':
        await remove_aga_help(ctx)
        return

    aga = args[0]
    if 'AGA' not in aga:
        aga = f"AGA{aga}"

    member = ctx.author
    user_id = member.id
    user_file = f'{_BURN_DATA.entrants_dir}/{user_id}.txt'
    with open(user_file, 'r') as fobj:
        data = fobj.readlines()[0].split(',')
    user_id = data[0]
    wallet = data[1]
    if aga in data[2:]:
        data.remove(aga)
        # overwrite the original file with a new one where the desired AGA
        # has been removed
        with open(user_file, 'w+') as fobj:
            fobj.write(','.join(data))
        await ctx.author.send(f'Removed {aga} from competitor list.')
        await ctx.message.add_reaction('\N{WHITE HEAVY CHECK MARK}')
    else:
        await ctx.author.send(f'{aga} is not registered.')

async def my_entries_help(ctx):
    msg = 'Algorillas Burnament Bot Help: **#my_entries**\n'
    msg += (
        "This bot is used to display your tournament entries"
    )
    await ctx.author.send(msg)
    msg +=(
        'Usage:\n' 
        '**#my_entries**\n'
    )
    await ctx.author.send(msg)

@bot.command('my_entries')
async def my_entries(ctx, *args):
    # if str(ctx.author) != 'cauchy69.APE#8518':
    #     await ctx.author.send(f'Command not available yet. Check back when registration opens!')
    #     return
    if len(args) != 0:
        await my_entries_help(ctx)
    member = ctx.author
    user_id = member.id
    user_file = f'{_BURN_DATA.entrants_dir}/{user_id}.txt'
    with open(user_file, 'r') as fobj:
        data = fobj.readlines()[0].split(',')

    user_id = data[0]
    wallet = data[1]
    wallet_str = f'{wallet[:4]}...{wallet[-4:]}'

    agas = data[2:]
    msg = f"**Registered wallet:** {wallet_str}\n"
    msg += '-'*10 + "**Competitors**" + '-'*10 + '\n'
    for aga in agas:
        aga_cut = _BURN_DATA.arc69_df[_BURN_DATA.arc69_df.unit_name == aga].iloc[0]
        msg += f"{aga_cut['name']}, Ranking: {aga_cut['rank']}/1000\n"
    await ctx.author.send(msg)
    await ctx.message.add_reaction('\N{WHITE HEAVY CHECK MARK}')


# @bot.command(name='showit')
async def get_img(ctx, aga):
    if 'AGA' not in aga:
        aga = f"AGA{aga}"
    image_dir = os.path.join(
        _BURN_DATA.project_dir,
        'images',
        'algorillas'
    )
    fname = f"{image_dir}/{aga}.png"

    return fname

async def aga_info_help(ctx):
    msg = 'Algorillas Burnament Bot Help: **#aga_info**\n'
    msg += (
        "This bot is used to retrieve information about any AGA.\n"
    )
    msg += 'Usage:\n'
    msg += '**#aga_info <aga number>**'
    await ctx.author.send(msg)
    msg = ''
    msg += '**Example:**\n'
    msg += '**#aga_info 123**\n'
    msg += 'This will print out information for AGA123,' \
           ' including the NFT image.'
    await ctx.author.send(msg)
    return

@bot.command(name='aga_info')
async def aga_info(ctx, aga):
    if aga == 'help':
        await aga_info_help(ctx)
        return
    if 'aga' not in aga.lower():
        aga = f"AGA{aga}"
    aga = aga.upper()
    aga = _BURN_DATA.arc69_df[_BURN_DATA.arc69_df.unit_name == aga].iloc[0]
    nontrait_cols = ['name','unit_name','asa', 'rank', 'rarity_score']
    trait_cols = [
        col for col in _BURN_DATA.arc69_df.columns if col not in nontrait_cols
    ]
    msg = ""
    for col, l in zip(nontrait_cols, ['Name','Unit Name','ASA', 'Rank','Rarity Score']):
        if l == 'Rarity Score':
            msg += f"**{l}**: {aga[col]:0.3f}\n"
        else:
            msg += f"**{l}**: {aga[col]}\n"

    await ctx.author.send(msg)
    msg = f"{'**Traits**'.center(50,'-')}\n"
    for col in trait_cols:
        if not isinstance(aga[col], str):
            continue
        msg += (
            f"**{col.capitalize()}**: {aga[col]} "
            f"({_BURN_DATA.trait_rarities[col][aga[col]]*1000:0.0f}/1000)\n"
        )
    await ctx.author.send(msg)
    fname = await get_img(ctx, aga['unit_name'])
    # LOG.info(fname)
    await ctx.author.send(file=discord.File(fname))
    await ctx.message.add_reaction('\N{WHITE HEAVY CHECK MARK}')


async def wallet_info_help(ctx):
    msg = 'Algorillas Burnament Bot Help: **#wallet_info**\n'
    msg += ('This bot is will return info on the AlgorillaArmy NFTs in the' 
           ' provided wallet. The provided information included listed NFTs. \n')
    msg += (
        'Usage:\n'
        '**#wallet_info** <wallet_address or NFD>\n'
        'If you are a registered participant, the wallet address argument is'
        ' optional.'
    )
    await ctx.author.send(msg)

@bot.command(name='wallet_info')
async def wallet_info(ctx, *args):
    if len(args) == 0:
        user_id, wallet, agas = await get_user_data(ctx)
    elif args[0] == 'help':
        await wallet_info_help(ctx)
        return
    else:
        wallet = args[0]

        if len(wallet) != 58:
            wallet = await lookup_nfd(ctx, wallet.lower())

    if wallet == "Not found":
        return
    try:
        df = _BURN_DATA.aga_holder_df.groupby('address').get_group(wallet)
    except KeyError as e:
        await ctx.author.send(
            f'{wallet[:4]}...{wallet[-4:]} has no AGAs'
        )
    await ctx.author.send(
        (f'{wallet[:4]}...{wallet[-4:]} holds {len(df)} '
         'AlgorillaArmy'
         )
    )
    messages = []
    msg = f"{'**AGA by Rank**'.center(50, '-')}\n"
    df = df.sort_values(by='rank', ascending=True)
    msg_length = 0
    j=0
    for i, row in df.iterrows():
        msg += f"{j + 1:0.0f}) {row['name']}, Ranking {row['rank']:0.0f}/1000\n"
        msg_length += len(msg)
        if len(msg)>1500:
            messages.append(msg)
            msg = ''
            current_length = msg_length
        j += 1
    msg_list_length = sum([len(val) for val in messages])
    if msg_list_length == 0:
        messages.append(msg)
    elif msg_list_length < msg_length:
        messages.append(msg)

    for msg in messages:
        await ctx.author.send(msg)
    await ctx.message.add_reaction('\N{WHITE HEAVY CHECK MARK}')

@bot.command(name='load_burnament')
async def load_burnament(ctx, *args):
    if str(ctx.author) != 'cauchy69.APE#8518':
        await ctx.author.send(f'{ctx.author} not authorized')
        return
    # _BURN_DATA.load_competitors(int(args[0]))
    _BURN_DATA.load_competitors()
    _BURN_DATA.initialize_bracket()
    msg = f"Burnament Loaded!\n"
    msg +=f"{'**Top 10 Seeds**'.center(50,'-')}\n"
    j = 0
    for i, row in _BURN_DATA.entrants.iterrows():
        if j == 10 or j > _BURN_DATA.entrants.shape[0]:
            break
        try:
            member = await bot.fetch_user(row["user"])
        except Exception as e:
            LOG.error(e)
            member = 'Unregistered'

        if isinstance(member, str):
            msg += (f"({row['seed']}) {row['name']}, "
                    f"**Rank:** {row['rank']}, **Holder:** {member}\n")
        else:
            msg += (f"({row['seed']}) {row['name']}, "
                    f"**Rank:** {row['rank']}, **Holder:** {member.mention}\n")
        j+=1
    await ctx.send(msg)
    msg = ''
    msg += f"{'**Tournament Rounds**'.center(50,'-')}\n"
    msg += '\n'.join(_BURN_DATA.round_names)
    await ctx.send(msg)


@bot.command(name='round_summary')
async def round_summary(ctx, *args):
    if str(ctx.author) != 'cauchy69.APE#8518':
        await ctx.send(f'{ctx.author} not authorized')
        return
    round_name = args[0]
    messages = await _BURN_DATA.print_round_summary(round_name, bot)
    # print(messages)
    for msg in messages:
        await ctx.send(msg)


@bot.command(name='winners_giveaway')
async def winners_giveaway(ctx, *args):
    if str(ctx.author) != 'cauchy69.APE#8518':
        await ctx.author.send(f'{ctx.author} not authorized')
        return
    round_name = args[0]
    N_winners = int(args[1])
    winner_list = _BURN_DATA.get_round_winners(round_name)
    # print(winner_list)
    messages = []
    msg = '-'*10 +\
          f'**{round_name} Winners Giveaway Entrants**' +\
          '-'*10 +'\n'
    msg_length = 0
    for winner in winner_list:
        member = await bot.fetch_user(winner['user'])
        msg += f"({winner['seed']}) {winner['name']}, {member.mention}\n"
        msg_length += len(msg)
        if len(msg) > 1000:
            messages.append(msg)
            msg = ''
            current_length = msg_length

    msg_list_length = sum([len(val) for val in messages])
    if msg_list_length == 0:
        messages.append(msg)
    elif msg_list_length < current_length:
        messages.append(msg)

    if messages:
        t = [await ctx.send(msg) for msg in messages]
    else:
        await ctx.send(msg)
    np.random.shuffle(winner_list)
    giveaway_winners = np.random.choice(
        len(winner_list), size=N_winners
    )
    if N_winners > 1:
        msg = '**Giveaway Winners:**\n'
        winners_file = (
            f"{_BURN_DATA.giveaway_dir}/"
            f"{round_name.replace(' ','_')}.txt"
        )
        with open(winners_file, 'w+') as fobj:
            for w in giveaway_winners:
                member = await bot.fetch_user(winner_list[w]['user'])
                msg += (
                    f"({winner_list[w]['seed']}) {winner_list[w]['name']}\n"
                )
                # msg += (
                #     f"({winner_list[w]['seed']}) {winner_list[w]['name']}, "
                #     f"{member.mention}\n"
                # )
                msg += f"Congrats {member.mention}, you won {_ALGO_PRIZE_AMOUNT} ALGO!\n"
                user_data = (
                    f"{winner_list[w]['name']},"
                    f"{winner_list[w]['user']},"
                    f"{winner_list[w]['wallet']}\n"
                )
                fobj.write(user_data)
        # await ctx.send(msg)
    else:
        w = winner_list[giveaway_winners[0]]
        member = await bot.fetch_user(w['user'])
        msg = '**Giveaway Winner:**\n'
        msg += (
                f"({w['seed']}) {w['name']}\n"
        )
        msg += f"Congrats {member.mention}, you won {_ALGO_PRIZE_AMOUNT} ALGO!\n"
        winners_file = (
            f"{_BURN_DATA.giveaway_dir}/"
            f"{round_name.replace(' ', '_')}.txt"
        )
        with open(winners_file, 'w+') as fobj:
            user_data = (
                f"{w['name']},"
                f"{w['user']},"
                f"{w['wallet']}\n"
            )
            fobj.write(user_data)
    await ctx.send(msg)


@bot.command(name='run_round')
async def run_round(ctx, *args):
    if str(ctx.author) != 'cauchy69.APE#8518':
        await ctx.author.send(f'{ctx.author} not authorized')
        return

    round_name = args[0]
    verbose = False

    if args[1] == 'verbose':
        verbose = True

    await ctx.send(f'Running {round_name}...')

    if _BURN_DATA.round_history is None:
        _BURN_DATA.round_history = {
            round_name: {'top': {}, 'bottom': {}}
        }
    elif round_name == 'Championship Round':
        _BURN_DATA.round_history[round_name] = {}

    else:
        _BURN_DATA.round_history[round_name] = {
            'top': {}, 'bottom':{}
        }

    if round_name == 'Championship Round':
        matches = _BURN_DATA.matchups['Championship Round']
        _BURN_DATA.round_history[round_name]['match'] = matches

        w = await pick_winner(
            ctx,
            matches[0]['unit_name'],
            matches[1]['unit_name'],
            verbose=verbose
        )
        winner = _BURN_DATA.entrants[_BURN_DATA.entrants.unit_name == w[-1]].iloc[0]
        msg = (
            '\N{PARTY POPPER}\N{PARTY POPPER}'
            f'**({winner["seed"]}) {winner["name"]} is the champion**'
            '\N{PARTY POPPER}\N{PARTY POPPER}'
        )

        await ctx.send(msg)
        # w1 = _BURN_DATA.entrants[_BURN_DATA.entrants.unit_name == winner].iloc[0]
        _BURN_DATA.round_history[round_name]['winner'] = winner

    else:
        for k, draw in zip([0, 1], ['top', 'bottom']):
            matches = _BURN_DATA.matchups[k]
            _BURN_DATA.round_history[round_name][draw]['matches'] = matches
            round_num = 0
            # while n_competitors >=2:
            round_winners = []
            for m in matches:
                w = await pick_winner(
                    ctx,
                    m[0]['unit_name'],
                    m[1]['unit_name'],
                    verbose=verbose
                )
                round_winners.append(w[-1])
                asyncio.sleep(2)
            round_winners = [
                _BURN_DATA.entrants[_BURN_DATA.entrants.unit_name == w].iloc[0]
                for w in round_winners
            ]

            _BURN_DATA.round_history[round_name][draw]['winners'] = \
                round_winners
            _BURN_DATA.matchups[k] = list(
                zip(round_winners[::2], round_winners[1::2]))
        if round_name == 'Semifinal Round':
            top_half_winner = \
            _BURN_DATA.round_history[round_name]['top']['winners'][0]

            bottom_half_winner = \
            _BURN_DATA.round_history[round_name]['bottom']['winners'][0]

            _BURN_DATA.matchups['Championship Round'] = [top_half_winner,
                                                         bottom_half_winner]
    _BURN_DATA.save_round(round_name)
    await ctx.send(f'Finished {round_name}')

@bot.command(name='fight')
async def fight(ctx, *args):
    if str(ctx.author) != 'cauchy69.APE#8518':
        await ctx.author.send(f'{ctx.author} not authorized')
        return
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
    info = _BURN_DATA.entrants[_BURN_DATA.entrants.unit_name == winner].iloc[0]
    await ctx.send(f'**{winner} is the champion**'.center(30, '\N{PARTY POPPER}'))
    await ctx.send(f'Congrats {info["user"]}'.center(10, '\N{PARTY POPPER}'))

@bot.command(name='battle')
async def pick_winner(ctx, aga1, aga2, verbose=True):
    if str(ctx.author) != 'cauchy69.APE#8518':
        await ctx.send(f'{ctx.author} not authorized')
        return

    if verbose:
        await ctx.send("-" * 10+'**FIGHT**'+'-'*10)

    if 'AGA' not in aga1 and 'BYE' not in aga1:
        aga1 = f'AGA{aga1}'

    if 'AGA' not in aga2 and 'BYE' not in aga2:
        aga2 = f'AGA{aga2}'
    try:
        aga1 = _BURN_DATA.entrants[_BURN_DATA.entrants.unit_name == aga1].iloc[0]
    except AttributeError as e:
        aga1 = _BURN_DATA.aga_holder_df[
            _BURN_DATA.aga_holder_df.unit_name == aga1
        ].iloc[0]

    try:
        aga2 = _BURN_DATA.entrants[_BURN_DATA.entrants.unit_name == aga2].iloc[0]
    except AttributeError as e:
        aga2 = _BURN_DATA.aga_holder_df[
            _BURN_DATA.aga_holder_df.unit_name == aga2
        ].iloc[0]

    s1_rs = aga1['rarity_score']
    s1_name = aga1['name']
    s1_unit_name = aga1['unit_name']
    try:
        s1_seed = aga1['seed']
    except KeyError:
        s1_seed = ''
    try:
        s1_user = aga1['user']
    except KeyError:
        s1_user = 'Unregistered'

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

    try:
        s2_user = aga2['user']
    except KeyError as e:
        s2_user = 'Unregistered'

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
        try:
            member = await bot.fetch_user(
                int(s1_user)
            )
            # user = member
        except Exception as e:
            user = s1_user
        # user = s1_user
        LOG.info(member)
    else:
        winner_seed = s2_seed_msg
        w = s2_unit_name
        try:
            member = await bot.fetch_user(
                int(s2_user)
            )
            # user = member
        except Exception as e:
            member = s2_user

        # user = s2_user
        LOG.info(member)
    if verbose:
        await ctx.send(f"**Draws**\n {a}")
        # await ctx.send("-" * 40)
        if isinstance(member, str):
            await ctx.send(
                f"**Winner:** {winner_seed} {winner} **Holder:** {member}\n"
            )
        else:
            await ctx.send(
                f"**Winner:** {winner_seed} {winner} **Holder:** {member.mention}\n"
            )
        fname = await get_img(ctx, w)
        await ctx.send(file=discord.File(fname))

    return results, entries, w


bot.run(_BOT_TOKEN)