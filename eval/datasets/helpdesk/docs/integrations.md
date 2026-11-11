# Integrations

## Calendar sync
Tallyhive can show task due dates in Google Calendar and Outlook. Sync is one-way: changes made in the calendar do not update the task. The calendar feed refreshes every 15 minutes.

## Slack
Install the Slack app from the Integrations page to receive task assignments in a channel of your choice. Use the command /tally add followed by a title to create a task without leaving Slack.

## Webhooks
Business plan workspaces can register up to 20 webhooks. Each webhook receives a JSON payload when a task is created, completed or deleted. Failed deliveries are retried five times with growing delays.

## Importing from other tools
Tallyhive imports CSV files exported from Trello, Asana and spreadsheets. Attachments are not carried over during an import, only titles, descriptions, due dates and assignees.
