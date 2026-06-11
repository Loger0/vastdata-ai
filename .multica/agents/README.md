# Multica Agent Instructions

This directory contains agent instruction files for the Multica eco-adapter squad.

## Delta Files

Files suffixed with `-delta.md` contain NEW sections to be inserted into existing agent instructions on the Multica platform. Each delta file has insertion instructions at the top.

## Deployment

To deploy updates:
1. Get current agent instructions: `multica agent get <agent-id> --output json`
2. Insert the delta content at the specified location
3. Deploy: `multica agent update <agent-id> --instructions "$(cat updated-instructions.md)"`
