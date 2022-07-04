from collections import Counter, defaultdict
import os

import discord
from discord.ext import commands


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

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command(name='at_info')
async def at_info(ctx, tourist):
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

    await ctx.send(msg)
    msg = f"{'**Traits**'.center(50,'-')}\n"
    for col in trait_cols:
        trait_rarirty = _AT_TRAIT_RARITIES[col][tourist[col]]
        # if tourist[col] == 'None':
        #     continue
        msg += f"**{col.capitalize()}**: {tourist[col]} ({trait_rarirty:0.2%})\n"

    await ctx.send(msg)

async def burn_help(ctx):
    msg = 'Alien Tourism Bot Help\n'
    msg += ('This bot is designed to burn tourist '
            'with a specific value for a given trait. If the value for the given'
            'trait contains a space, then it will need to be enclosed with double quotes.\n')
    msg += ('Given a trait, e.g. Single Outfit, '
            'and a specific value, e.g. Tree hawaiian, and a number, this bot'
            'will burn all Tourist with the Tree hawaiian value for Single'
            ' Outfit while keeping the tourist matching the number provided.')
    await ctx.send(msg)
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
    await ctx.send(msg)

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
        await ctx.send(
            (f'**{tourist_to_keep} not found.** '
             'Double check the input name.')
        )
        return
    try:
        keep_value = keep[trait].iloc[0]
    except KeyError as e:
        await ctx.send(
            (f"**Trait {trait} does not exist.** "
             f"Double check the input trait."
             )
        )
        return
    else:
        if keep_value.lower() != value.lower():
            await ctx.send(
                (f'**{tourist_to_keep} does not have desired trait value.** '
                 'Double check the input trait value.')
            )
            return

    trait_cut = _AT_DF[_AT_DF[trait] == value]
    if trait_cut.empty:
        await ctx.send(
            (f'**Tourists with {trait}={value} not found.** '
            'Double check the input trait.')
        )
        return

    msg = f"Burning {value} {trait.lower()}"
    await ctx.send(msg)
    await ctx.send(f'Keeping Tourist: {tourist_to_keep}')
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
        await ctx.send(msg)
    else:
        msg = (f'\N{FIRE}\N{FIRE}\N{FIRE}\N{FIRE}\n'
               f'Burned {aliens_to_burn.shape[0]} Tourists with trait'
               f'\n\N{FIRE}\N{FIRE}\N{FIRE}\N{FIRE}')
        await ctx.send(msg)

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
    await ctx.send(msg)

    top_ten_before = '**Top 10 Tourist Before Burn**\n'
    j=0
    for i, row in _AT_DF.iterrows():
        if j == 10:
            break
        else:
            top_ten_before += f"{j + 1:0.0f}) {row['name']}\n"
        j+=1
    await ctx.send(top_ten_before)


    msg = f"{'**After Burn**'.center(50,'-')}\n"
    tourist_after = burn_df[burn_df.unit_name == keep.unit_name.iloc[0]].iloc[0]
    for col, l in zip(nontrait_cols, ['Name','ASA', 'Rank','Rarity Score']):
        if l == 'Rarity Score':
            msg += f"**{l}**: {tourist_after[col]:0.2f}\n"
        else:
            msg += f"**{l}**: {tourist_after[col]}\n"

    await ctx.send(msg)

    top_ten_after = '**Top 10 Tourist After Burn**\n'
    burn_df = burn_df.sort_values(by='rank', ascending=True)
    j=0
    for i, row in burn_df.iterrows():
        if j == 10:
            break
        else:
            top_ten_after += f"{j + 1:0.0f}) {row['name']}\n"
        j+=1
    await ctx.send(top_ten_after)


async def burn_list_help(ctx):
    msg = 'Alien Tourism Bot Help\n'
    msg += 'This bot is designed to burn a list of tourist specified by the user.\n'
    msg += 'The list should be a set of space-separated numbers.\n'
    msg += 'The final number in the list corresponds to the alien you want to keep.'
    await ctx.send(msg)
    msg = ''
    msg += 'Usage:\n'
    msg += '**$burn_list <N1> <N2> <N3> ... <N>\n**'
    msg += 'Examples\n'+f"{'-'*15}\n"
    msg += '**$burn_list 1 2 3 4 5 6\n**'
    msg += 'This will burn Tour1, Tour2, Tour3, Tour4, Tour5, while keeping Tour6\n'
    msg += '**$burn_list 844 1371 2412 3932 2232\n**'
    msg += 'This will burn Tour844, Tour1371, Tour2412, Tour3932, while keeping Tour2232.'
    await ctx.send(msg)
    return ctx

@bot.command(name='burn_list')
async def burn_list(ctx, *args):
    if args[0] == 'help':
        await burn_list_help(ctx)
        return

    burn_list = [f'Tour{val}' for val in args[:-1]]
    tourist_to_keep = f'Tour{args[-1]}'
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
        await ctx.send(
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
        await ctx.send(msg)
    else:
        msg = ('Burning Tourists\n'+"\n".join(burn_list) +
               '\n\N{FIRE}\N{FIRE}\N{FIRE}\N{FIRE}')
        await ctx.send(msg)



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
    await ctx.send(msg)


    msg = f"{'**After Burn**'.center(50,'-')}\n"
    tourist_after = burn_df[burn_df.unit_name == keep.unit_name.iloc[0]].iloc[0]
    for col, l in zip(nontrait_cols, ['Name','ASA', 'Rank','Rarity Score']):
        if l == 'Rarity Score':
            msg += f"**{l}**: {tourist_after[col]:0.2f}\n"
        else:
            msg += f"**{l}**: {tourist_after[col]}\n"
    await ctx.send(msg)




_AT_TRAIT_RARITIES = compute_trait_rarities(_AT_DF, attributes, burn=False)
bot.run(_BOT_TOKEN)