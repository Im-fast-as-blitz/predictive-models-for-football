#!/bin/bash

BASE_URL="http://89.169.183.3:8000"

curl -s $BASE_URL/seasons

curl -s $BASE_URL/teams

curl -X POST $BASE_URL/forward \
  -H "Content-Type: application/json" \
  -d '{"HomeTeam": "Charlton", "AwayTeam": "Everton", "season": "2001-2002"}'

curl -X POST $BASE_URL/forward \
  -H "Content-Type: application/json" \
  -d '{"HomeTeam": "Arsenal", "AwayTeam": "Chelsea"}'

