import asyncio
from collections import Counter, defaultdict
import glob
import logging
import os
import time

import discord
from discord.ext import commands
import pandas as pd
import numpy as np
from burnament_helper import BurnamentData

# Set up the style for logging output
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s')

# instantiate the logger
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)


bot = commands.Bot(command_prefix='>')
_BOT_TOKEN = os.environ['AIRDROP']

# initialize the burnament data class
_BURN_DATA = BurnamentData()
_BURN_DATA.compute_trait_rarities()

_QUALIFIED_HOLDERS = pd.read_csv(
    './holders_airdrop_registration.csv',
    header=0,
    index_col=None
)

_N_SHUFFLES = 0


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

@bot.command(name='airdrop_shuffle')
# @commands.has_role('ADMIN')
async def airdrop_shuffle(ctx, *args):
    global _N_SHUFFLES
    if str(ctx.author) != 'cauchy69.APE#8518':
        await ctx.send(f'{ctx.author} is an unauthorized user')
        return

    discord_handle = args[0]
    try:
        member = await commands.MemberConverter().convert(ctx, discord_handle)
    except Exception as e:
        member = discord_handle
    draw = _BURN_DATA.unreleased_aga_df.sample(1)
    _N_SHUFFLES += 1

    fname = await get_img(ctx, draw['unit_name'].iloc[0])
    if isinstance(member, str):
        msg = f'Shuffle draw for {member}\n'
    else:
        msg = f'Shuffle draw for {member.mention}\n'
    msg += (f'{draw["name"].iloc[0]}, '
            f'Rank: {draw["rank"].iloc[0]}/1000, '
            f'ASA ID: {draw["asa"].iloc[0]}\n'
            )
    await ctx.send(msg)
    await ctx.send(file=discord.File(fname))
    await ctx.send(
        (f'The number of draws remaining is: '
        f'{_QUALIFIED_HOLDERS.shape[0] - _N_SHUFFLES:0.0f}\n')
    )
    msg = '\N{PARTY POPPER}'*5

    await ctx.send(msg)
    try:
        await member.send(
            (f'Please opt-in to {draw["name"].iloc[0]}\n '
             f'**ASA ID: {draw["asa"].iloc[0]}\n**'
             'Once all of the shuffle draws have concluded, '
             'we will start distribution.\n')
        )
    except Exception as e:
        LOG.error('DM Failed')

    LOG.info('Removing shuffle AGA from available list')
    _BURN_DATA.unreleased_aga_df.drop(index=draw.index, inplace=True)
    LOG.info(f'AGA remaining: {_BURN_DATA.unreleased_aga_df.shape[0]}')
    user_data = _QUALIFIED_HOLDERS[
        _QUALIFIED_HOLDERS['discord']==discord_handle
    ].iloc[0]
    if not os.path.exists('shuffle_draws.csv'):
        with open('shuffle_draws.csv', 'w+') as fobj:
            fobj.write(
                (
                    'discord,address,unit_name,asa,creator_address\n'
                    f'{user_data["discord"]},'
                    f'{user_data["address"]},'
                    f'{draw["unit_name"].iloc[0]},'
                    f'{draw["asa"].iloc[0]},'
                    f'{draw["creator_address"].iloc[0]}\n'
                )
            )
    else:
        with open('shuffle_draws.csv', 'a') as fobj:
            fobj.write(
                (
                    f'{user_data["discord"]},'
                    f'{user_data["address"]},'
                    f'{draw["unit_name"].iloc[0]},'
                    f'{draw["asa"].iloc[0]},'
                    f'{draw["creator_address"].iloc[0]}\n'
                )
            )
    return


if __name__ == "__main__":
    bot.run(_BOT_TOKEN)