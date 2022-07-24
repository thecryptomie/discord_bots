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

# Set up the style for logging output
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s')

_BOT_TOKEN = os.environ['REBIRTH']
# instantiate the logger
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

with open('rebirthed_apes.txt', 'r') as fobj:
    rebirthed_apes = fobj.readlines()
    rebirthed_apes = ["APE#"+val.strip('\n') for val in rebirthed_apes]

bot = commands.Bot(command_prefix='>')

@bot.command(name='check')
async def check(ctx, *args):
    ape = args[0]
    if 'APE' not in ape:
        ape = 'APE#'+ape

    if ape in rebirthed_apes:
        msg = f'{ape} has been rebirthed'
    else:
        msg = f"You and {ape} have a lot in common, you're both virgins"

    await ctx.send(msg)
    return

if __name__ == "__main__":
    bot.run(_BOT_TOKEN)

