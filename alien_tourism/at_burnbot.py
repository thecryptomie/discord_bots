from collections import Counter, defaultdict
import os

import discord
from discord.ext import commands

import requests
import pandas as pd
intents = discord.Intents.default()
intents.messages = True

client = discord.Client(intents=intents)

_BOT_TOKEN = os.environ['ALIENTOURISM']

bot = commands.Bot(command_prefix='$')

_AT_DF = pd.read_csv(
    os.path.expanduser('~/discord_bots/arc69/alien_tourism_with_rank_arc69.csv'),
    header=0,
    index_col=None
)
_HOLDERS_DF = pd.read_csv(
    os.path.expanduser('~/discord_bots/holder_data/alien_tourism.csv'),
    header=0,
    index_col=None
)


_HOLDERS_DF = pd.merge(
    _HOLDERS_DF,
    _AT_DF,
    on='asa',
    how='inner',
    suffixes=['','_1']
)
# Drop repeat columns
_HOLDERS_DF = _HOLDERS_DF.drop(columns=['name_1','unit_name_1'])
attributes = [
    'Background',
    'Eyes',
    'Pose',
    'Skin',
    'Hair',
    'Shirt',
    'Jacket',
    'Single Outfit',
    'Necklace',
    'Mask',
    'Foam Finger',
    'Hat',
    'Glasses'
]
attributes = [val.lower() for val in attributes]

_ROUND_WINNERS = defaultdict(list)

def compute_trait_rarities(arc69, attributes, burn=True):
    trait_rarities = {}
    for attr in attributes:
        trait_rarities[attr] = {}
        for grp, df in arc69.groupby(attr):
            trait_rarities[attr][grp] = len(df) / len(arc69)
    rarity_score = []
    for i, row in arc69.iterrows():
        score = 0
        row = row.dropna()
        for attr in attributes:
            if attr in row.index:
                score += 1 / trait_rarities[attr][row[attr]]
        rarity_score.append(score)

    arc69['rarity_score'] = rarity_score
    arc69.sort_values(by='rarity_score', inplace=True, ascending=False)
    arc69['rank'] = [i + 1 for i in range(len(arc69))]
    if burn:
        return arc69, trait_rarities
    else:
        return trait_rarities


@bot.command(name='get_help')
async def get_help(ctx):
    bot_commands = [
        '**$wallet_info**',
        '**$wallet_list**',
        '**$at_info**',
        '**$at_list**',
        '**$burn**',
        '**$burn_list**'
    ]
    msg = 'Alien Tourism Bot Help\n'
    msg += 'The available commands are,\n'
    msg += '\n'.join(bot_commands)
    await ctx.send(msg)
    msg = 'Each command has its own set of information accessible through help.\n'
    msg += 'Accessing the help menu for a given command\n **$<command_name> help**\n'
    msg += (
            'For example, need help using the **$wallet_list** command?'
            ' Just type **$wallet_list help**.'
    )
    await ctx.send(msg)


@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

async def holder_traits(df):
    nontrait_cols = ['name', 'asa', 'rank', 'rarity_score', 'unit_name']
    trait_cols = [
        col for col in _AT_DF.columns if col not in nontrait_cols
    ]
    trait_data = defaultdict(list)
    for col in trait_cols:
        trait_data[col]
        for val in df[col]:
            if val == 'none':
                continue
            trait_data[col].append(val)

    trait_counts = {}
    for key, val in trait_data.items():
        trait_counts[key] = Counter(val)

    return trait_counts
#
# @bot.command(name='register')
# async def register(ctx, wallet):

async def wallet_info_help(ctx):
    msg = 'Alien Tourism Bot Help: **wallet_info**\n'
    msg += ('This bot is will return statistics on the Alien Tourism NFTS' 
           ' provided wallet. All statistics included listed NFTs. \n')
    msg += (
            'Stats reported:\n '
            'Number of NFTs held\n'
            'Top five rarest Tourists\n'
            'Lists of traits that you collect\n')
    await ctx.author.send(msg)
    msg = 'Usage:\n'
    msg += "**$wallet_info <wallet address or NFD>**\n"
    msg += 'Example:'
    msg += '$wallet_info UWVYY2WRT7CCRPWNYUUKQ6HCO7JYTGH6TNQIJIJV6X2PG74UML2HGU5BAA'
    await ctx.author.send(msg)


@bot.command(name='wallet_info')
async def wallet_info(ctx, *args):
    wallet = args[0]
    if wallet == 'help':
        await wallet_info_help(ctx)

    if len(wallet) != 58:
        wallet = await lookup_nfd(ctx, wallet)

    df = _HOLDERS_DF.groupby('address').get_group(wallet)
    await ctx.author.send(
        (f'{wallet[:4]}...{wallet[-4:]} holds {len(df)} '
         'Tourists'
         )
    )
    # top_five = '**Top 5 Tourist by Rank**'.center(25, '-') +'\n'
    top_five = f"{'**Top 5 Tourist by Rank**'.center(50, '-')}\n"
    df = df.sort_values(by='rank', ascending=True)
    j=0
    for i, row in df.iterrows():
        if j == 5:
            break
        else:
            top_five += f"{j + 1:0.0f}) {row['name']} (Rank {row['rank']:0.0f})\n"
        j+=1
    await ctx.author.send(top_five)

    nontrait_cols = ['name', 'asa', 'rank', 'rarity_score', 'unit_name']
    trait_cols = [
        col for col in _AT_DF.columns if col not in nontrait_cols
    ]
    trait_data = defaultdict(list)
    for col in trait_cols:
        trait_data[col]
        for val in df[col]:
            if val == 'none':
                continue
            trait_data[col].append(val)
    trait_counts = {}
    for key, val in trait_data.items():
        trait_counts[key] = Counter(val)

    msg = f"{'**Most Collected Traits**'.center(50,'-')}\n"
    for key, val in trait_counts.items():
        print(key, val.most_common(1))
        most_common = val.most_common(1)
        if most_common:
            mc = val.most_common(1)[0]
        msg += f"**{key.capitalize()}**: {mc[0]} ({mc[1]:0.0f}x)\n"

    await ctx.author.send(msg)

    # return trait_counts

async def at_list_help(ctx):
    """Help function for at_list

    Parameters
    ----------
    ctx

    Returns
    -------

    """
    msg = 'Alien Tourism Bot Help: **$at_list**\n'
    msg += (
        'This bot is designed to generate a list of tourists with the'
        ' specified traits. The inputs are a series of trait/value pairs '
        'and the number of the tourist you want to keep (e.g. 69 for Tour69). '
        'This list can then be copy and pasted into'
        ' **$burn_list** command to see how burning affects the rarity of '
        'the tourist you want to keep.'
    )
    await ctx.author.send(msg)

    msg = ''
    msg += 'Usage:\n'
    msg += '**$at_list <trait1> <value1> <trait2> <value2> <tourist_to_keep>\n**'
    msg += 'Examples\n' + f"{'-' * 15}\n"
    msg += '**$at_list hat "viking helmet" jacket "illuminated leather" 2657\n**'
    msg += (
        'This will generate a list of tourists with either a viking helmet'
        ' for a hat or an illuminated leather jacket. The last tourists in'
        ' the list will be Tour2657'
    )
    await ctx.author.send(msg)

@bot.command(name='at_list')
async def at_list(ctx, *args):
    """Generate a list of tourist with the given traits

    Parameters
    ----------
    ctx
    tourist

    Returns
    -------

    """
    if args[0] == 'help':
        await at_list_help(ctx)
        return
    elif len(args[:-1]) % 2 != 0:
        traits = args[::2]
        values = args[1::2]
        keep = None
        # await ctx.author.send('Invalid number of arguments. Check $at_list help for more')
        # return
    else:
        traits = args[::2]
        values = args[1::2]
        # try:
        #     int(args[-1])
        # except Exception as e:
        #     keep = None
        # else:
        if 'tour' in args[-1].lower():
            keep = args[-1]
        else:
            keep = 'Tour'+args[-1]
    trait_cuts = []
    for i, (t, v) in enumerate(zip(traits, values)):
        trait_cuts.append(_AT_DF[_AT_DF[t.lower()] == v.lower()])

    trait_cut = pd.concat(trait_cuts, axis=0)
    tourists = list(trait_cut['unit_name'].unique())
    # make the tourist we want to keep be at the end
    if keep is not None:
        tourists.remove(keep)
        tourists.append(keep)
    tourist_list = " ".join(tourists)
    await ctx.author.send(tourist_list)


async def wallet_list_help(ctx):
    """Help function for at_list

    Parameters
    ----------
    ctx

    Returns
    -------

    """
    msg = 'Alien Tourism Bot Help: **$wallet_list**\n'
    msg += (
        'This bot is designed to generate a list of tourists with the'
        ' specified traits from the user provided wallet. '
        'The inputs are a series of trait/value pairs '
        'and the number of the tourist you want to keep (e.g. 69 for Tour69). '
        'This list can then be copy and pasted into'
        ' **$burn_list** command to see how burning affects the rarity of '
        'the tourist you want to keep. The **<tourist_to_keep>** argument is '
        'optional. This is useful if you are interested in seeing what traits'
        ' are still held in the creators wallet.'
    )
    await ctx.author.send(msg)

    msg = ''
    msg += 'Usage:\n'
    msg += '**$wallet_list <trait1> <value1> <trait2> <value2> <tourist_to_keep> <wallet_address or NFD>\n**'
    msg += 'Examples\n' + f"{'-' * 15}\n"
    msg += '**$wallet_list hat "viking helmet" UWVYY2WRT7CCRPWNYUUKQ6HCO7JYTGH6TNQIJIJV6X2PG74UML2HGU5BAA\n**'
    msg += (
        'This will generate a list of tourists with a viking helmet hat held'
        'in the provided wallet (the creators address).\n'
    )
    msg += '**$wallet_list hat "viking helmet" 950 UWVYY2WRT7CCRPWNYUUKQ6HCO7JYTGH6TNQIJIJV6X2PG74UML2HGU5BAA\n**'
    msg += (
        'This will generate a list of tourists with a viking helmet hat held '
        'in the provided wallet (the creators address). The last tourist in this'
        ' list will be Tour950.'
    )

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
        await ctx.author.send(f'{nfd} could not be resolved')
        wallet = 'Not found'
    return wallet

@bot.command(name='wallet_list')
async def wallet_list(ctx, *args):
    """Generate a list of tourist with the given traits

    Parameters
    ----------
    ctx
    tourist

    Returns
    -------

    """

    if args[0] == 'help':
        await wallet_list_help(ctx)
        return
    elif len(args[:-2]) % 2 != 0:
        traits = args[:-1:2]
        values = args[1:-1:2]
        keep = None
        wallet = args[-1]
    else:
        traits = args[:-2:2]
        values = args[1:-2:2]
        if 'tour' in args[-2].lower():
            keep = args[-2]
        else:
            keep = 'Tour'+args[-2]
        wallet = args[-1]

    if len(wallet) != 58:
        wallet = await lookup_nfd(ctx, wallet)

    wallet_cut = _HOLDERS_DF[_HOLDERS_DF['address'] == wallet]
    trait_cuts = []
    for i, (t, v) in enumerate(zip(traits, values)):
        trait_cuts.append(wallet_cut[wallet_cut[t.lower()] == v.lower()])

    trait_cut = pd.concat(trait_cuts, axis=0)
    tourists = list(trait_cut['unit_name'].unique())
    # make the tourist we want to keep be at the end
    if keep is not None:
        tourists.remove(keep)
        tourists.append(keep)
    tourist_list = " ".join(tourists)
    await ctx.author.send(tourist_list)


@bot.command(name='at_info')
async def at_info(ctx, tourist):
    if tourist == 'help':
        await at_info_help(ctx)
    if 'Tour' not in tourist:
        tourist = f'Tour{tourist}'
    tourist = _AT_DF[_AT_DF.unit_name == tourist].iloc[0]
    nontrait_cols = ['name','asa', 'rank', 'rarity_score','unit_name']
    trait_cols = [
        col for col in _AT_DF.columns if col not in nontrait_cols
    ]
    msg = ""
    for col, l in zip(nontrait_cols, ['Name','ASA', 'Rank','Rarity Score']):
        if l == 'Rarity Score':
            msg += f"**{l}**: {tourist[col]:0.2f}\n"
        else:
            msg += f"**{l}**: {tourist[col]}\n"

    await ctx.author.send(msg)
    msg = f"{'**Traits**'.center(50,'-')}\n"
    for col in trait_cols:
        trait_rarirty = _AT_TRAIT_RARITIES[col][tourist[col]]
        if tourist[col] == 'none':
            continue
        msg += f"**{col.capitalize()}**: {tourist[col]} ({trait_rarirty:0.2%})\n"
    await ctx.author.send(msg)

async def at_info_help(ctx):
    """Help function for at_list

    Parameters
    ----------
    ctx

    Returns
    -------

    """
    msg = 'Alien Tourism Bot Help: **$at_info**\n'
    msg += (
        'This returns information (rank, rarity score, traits)'
        ' about the provided tourists.'
    )
    await ctx.author.send(msg)

    msg = ''
    msg += 'Usage:\n'
    msg += '**$at_info <tourist_number>\n**'
    msg += 'Examples\n' + f"{'-' * 15}\n"
    msg += '**$at_info 2657\n**'
    msg += (
        'This will return information about Tour2657.'
    )
    await ctx.author.send(msg)

async def burn_help(ctx):
    msg = 'Alien Tourism Bot Help: **$burn**\n'
    msg += ('This bot is designed to mimic the burning of tourists '
            'with a specific value for a given trait. '
            'If the value for the given trait contains a space, '
            'then it will need to be enclosed with double quotes.\n')
    msg += ('Given a trait, e.g. Single Outfit, '
            'and a specific value, e.g. Tree hawaiian, and a number, it'
            'will burn all Tourists with the Tree hawaiian value for Single'
            ' Outfit, while keeping the tourist matching the number provided.')
    await ctx.author.send(msg)
    msg = ''
    msg += 'Usage:\n'
    msg += '**$burn <trait> <value> <tourist_to_keep>\n**'
    msg += 'Examples\n' + f"{'-' * 15}\n"
    msg += '**$burn hat "viking helmet" 2232\n**'
    msg += ('This will burn all the viking '
            'helmet tourist with the exception of Tour2232. '
            'Note the use of double quotes.\n')
    msg += '**$burn skin gecko 3247\n**'
    msg += 'This will burn all of the gecko skins with the exception of Tour 3247'
    await ctx.author.send(msg)

@bot.command(name='burn')
async def burn(ctx, *args):
    if args[0] == 'help':
        await burn_help(ctx)
        return
    trait = args[0]
    value = args[1]
    if 'Tour' in args[2]:
        tourist_to_keep = args[2]
    else:
        tourist_to_keep = 'Tour'+args[2]

    random_burn = 0

    if len(args) == 4:
        random_burn = int(args[3])

    keep = _AT_DF[_AT_DF['unit_name'] == tourist_to_keep]
    if keep.empty:
        await ctx.author.send(
            (f'**{tourist_to_keep} not found.** '
             'Double check the input name.')
        )
        return
    try:
        keep_value = keep[trait].iloc[0]
    except KeyError as e:
        await ctx.author.send(
            (f"**Trait {trait} does not exist.** "
             f"Double check the input trait."
             )
        )
        return
    else:
        if keep_value.lower() != value.lower():
            await ctx.author.send(
                (f'**{tourist_to_keep} does not have desired trait value.** '
                 'Double check the input trait value.')
            )
            return

    trait_cut = _AT_DF[_AT_DF[trait] == value]
    if trait_cut.empty:
        await ctx.author.send(
            (f'**Tourists with {trait}={value} not found.** '
            'Double check the input trait.')
        )
        return

    msg = f"Burning {value} {trait.lower()}"
    await ctx.author.send(msg)
    await ctx.author.send(f'Keeping Tourist: {tourist_to_keep}')
    aliens_to_burn = trait_cut.drop(index=keep.index)

    burn_df = _AT_DF.drop(index=aliens_to_burn.index)
    # Burn an additional number of tourists
    if random_burn > 0:
        sample_df = burn_df[burn_df['unit_name'] != tourist_to_keep]
        random_drops = sample_df.sample(random_burn)
        burn_df = burn_df.drop(index=random_drops.index)

    burn_df, trait_rarities = compute_trait_rarities(
        burn_df,
        attributes,
        burn=True
    )
    if random_burn:
        msg = (f'\N{FIRE}\N{FIRE}\N{FIRE}\N{FIRE}\n'
               f'Burned {aliens_to_burn.shape[0]} with '
               f'{trait.lower()} {value.lower()} '
               f'plus {random_burn} extra tourists\n'
               f'\N{FIRE}\N{FIRE}\N{FIRE}\N{FIRE}')
        await ctx.author.send(msg)
    else:
        msg = (f'\N{FIRE}\N{FIRE}\N{FIRE}\N{FIRE}\n'
               f'Burned {aliens_to_burn.shape[0]} Tourists with trait'
               f'\n\N{FIRE}\N{FIRE}\N{FIRE}\N{FIRE}')
        await ctx.author.send(msg)

    nontrait_cols = ['name', 'asa', 'rank', 'rarity_score', 'unit_name']
    trait_cols = [
        col for col in _AT_DF.columns if col not in nontrait_cols
    ]

    tourist_before = _AT_DF[_AT_DF.unit_name == keep.unit_name.iloc[0]].iloc[0]
    msg = f"{'**Before Burn**'.center(50,'-')}\n"
    for col, l in zip(nontrait_cols, ['Name','ASA', 'Rank','Rarity Score']):
        if l == 'Rarity Score':
            msg += f"**{l}**: {tourist_before[col]:0.2f}\n"
        else:
            msg += f"**{l}**: {tourist_before[col]}\n"
    await ctx.author.send(msg)

    # top_ten_before = '**Top 10 Tourist Before Burn**\n'
    # j=0
    # for i, row in _AT_DF.iterrows():
    #     if j == 10:
    #         break
    #     else:
    #         top_ten_before += f"{j + 1:0.0f}) {row['name']}\n"
    #     j+=1
    # await ctx.author.send(top_ten_before)


    msg = f"{'**After Burn**'.center(50,'-')}\n"
    tourist_after = burn_df[burn_df.unit_name == keep.unit_name.iloc[0]].iloc[0]
    for col, l in zip(nontrait_cols, ['Name','ASA', 'Rank','Rarity Score']):
        if l == 'Rarity Score':
            msg += f"**{l}**: {tourist_after[col]:0.2f}\n"
        else:
            msg += f"**{l}**: {tourist_after[col]}\n"

    await ctx.author.send(msg)

    # top_ten_after = '**Top 10 Tourist After Burn**\n'
    # burn_df = burn_df.sort_values(by='rank', ascending=True)
    # j=0
    # for i, row in burn_df.iterrows():
    #     if j == 10:
    #         break
    #     else:
    #         top_ten_after += f"{j + 1:0.0f}) {row['name']}\n"
    #     j+=1
    # await ctx.author.send(top_ten_after)


async def burn_list_help(ctx):
    msg = 'Alien Tourism Bot Help: **$burn_list**\n'
    msg += 'This bot is designed to burn a list of tourist specified by the user.\n'
    msg += 'The list should be a set of space-separated numbers.\n'
    msg += 'The final number in the list corresponds to the alien you want to keep.\n'
    msg += 'There must be at least two numbers passed.'
    await ctx.author.send(msg)
    msg = ''
    msg += 'Usage:\n'
    msg += '**$burn_list <N1> <N2> <N3> ... <N>\n**'
    msg += 'Examples\n'+f"{'-'*15}\n"
    msg += '**$burn_list 1 2 3 4 5 6\n**'
    msg += 'This will burn Tour1, Tour2, Tour3, Tour4, Tour5, while keeping Tour6\n'
    msg += '**$burn_list 844 1371 2412 3932 2232\n**'
    msg += 'This will burn Tour844, Tour1371, Tour2412, Tour3932, while keeping Tour2232.'
    await ctx.author.send(msg)
    return ctx

@bot.command(name='burn_list')
async def burn_list(ctx, *args):
    if args[0] == 'help':
        await burn_list_help(ctx)
        return
    if len(args) < 2:
        await ctx.author.send(
            ('Need to pass at least two tourist numbers! '
             'Try **$burn_list help** for more details')
        )
        return

    burn_list = []
    for val in args[:-1]:
        if 'tour' in val.lower():
            burn_list.append(val.capitalize())
        else:
            burn_list.append(f'Tour{val}')
    # burn_list = [f'Tour{val}' for val in args[:-1]]
    if 'tour' not in args[-1].lower():
        tourist_to_keep = f'Tour{args[-1]}'
    else:
        tourist_to_keep = args[-1].capitalize()

    keep = _AT_DF[_AT_DF['unit_name'] == tourist_to_keep]
    random_burn = 0
    for i, tourist in enumerate(burn_list):
        if i == 0:
            aliens_to_burn = _AT_DF[_AT_DF['unit_name'] == tourist]
        else:
            aliens_to_burn = pd.concat(
                [aliens_to_burn,_AT_DF[_AT_DF['unit_name'] == tourist]]
            )

    if tourist_to_keep in burn_list:
        await ctx.author.send(
            'You burnt the tourist you want to keep! Check your list'
        )
        return
    burn_df = _AT_DF.drop(index=aliens_to_burn.index)
    # Burn an additional number of tourists
    if random_burn > 0:
        sample_df = burn_df[burn_df['unit_name'] != tourist_to_keep]
        random_drops = sample_df.sample(random_burn)
        burn_df = burn_df.drop(index=random_drops.index)

    burn_df, trait_rarities = compute_trait_rarities(
        burn_df,
        attributes,
        burn=True
    )
    if random_burn >0:
        msg = (f'\N{FIRE}\N{FIRE}\N{FIRE}\N{FIRE}\n'
               f'Burned {aliens_to_burn.shape[0]} tourists'
               f'plus {random_burn} extra tourists\n'
               f'\N{FIRE}\N{FIRE}\N{FIRE}\N{FIRE}')
        await ctx.author.send(msg)
    else:
        msg = ('Burning Tourists\n'+"\n".join(burn_list) +
               '\n\N{FIRE}\N{FIRE}\N{FIRE}\N{FIRE}')
        await ctx.author.send(msg)

    nontrait_cols = ['name', 'asa', 'rank', 'rarity_score', 'unit_name']
    trait_cols = [
        col for col in _AT_DF.columns if col not in nontrait_cols
    ]

    tourist_before = _AT_DF[_AT_DF.unit_name == keep.unit_name.iloc[0]].iloc[0]
    msg = f"{'**Before Burn**'.center(50,'-')}\n"
    for col, l in zip(nontrait_cols, ['Name','ASA', 'Rank','Rarity Score']):
        if l == 'Rarity Score':
            msg += f"**{l}**: {tourist_before[col]:0.2f}\n"
        else:
            msg += f"**{l}**: {tourist_before[col]}\n"
    await ctx.author.send(msg)


    msg = f"{'**After Burn**'.center(50,'-')}\n"
    tourist_after = burn_df[burn_df.unit_name == keep.unit_name.iloc[0]].iloc[0]
    for col, l in zip(nontrait_cols, ['Name','ASA', 'Rank','Rarity Score']):
        if l == 'Rarity Score':
            msg += f"**{l}**: {tourist_after[col]:0.2f}\n"
        else:
            msg += f"**{l}**: {tourist_after[col]}\n"
    await ctx.author.send(msg)




_AT_TRAIT_RARITIES = compute_trait_rarities(_AT_DF, attributes, burn=False)
bot.run(_BOT_TOKEN)