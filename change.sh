#!/bin/sh

git filter-branch -f --env-filter '

ad="$GIT_AUTHOR_DATE"
cd="$GIT_COMMITTER_DATE"

ADARR=($ad)
ad=${ADARR[0]}

ad=$(gdate -d "$ad")
cd=$ad

echo "before: $ad"

ad=$(gdate -d "$ad + 1200 day")
cd=$(gdate -d "$cd + 1200 day")

echo "after: $ad"

export GIT_AUTHOR_DATE="$ad"
export GIT_COMMITTER_DATE="$cd"
'
