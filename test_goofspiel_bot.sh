RUNCMD=./my_bot

echo "{\"ping\": 1234}" | $RUNCMD

echo "{\"player-cards\": {\"opponent\": [1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13], \"me\": [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13]}, \"trophy-cards\": [1, 3, 4, 5, 7, 8, 9, 10, 12, 13], \"current-trophy\": 11, \"history\": [{\"trophy\": 6, \"opponent\": 6, \"me\": 8}, {\"trophy\": 2, \"opponent\": 2, \"me\": 4}]}" | $RUNCMD


