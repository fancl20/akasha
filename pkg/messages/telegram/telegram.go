package telegram

import (
	"context"
	"fmt"
	"time"

	"github.com/PaulSonOfLars/gotgbot/v2"
	"github.com/fancl20/akasha/pkg/messages"
	"github.com/fancl20/akasha/pkg/notes"
)

type Channels struct {
	bot      *gotgbot.Bot
	channels map[int64]*Channel
}

func NewChannels(token string) (*Channels, error) {
	bot, err := gotgbot.NewBot(token, nil)
	if err != nil {
		return nil, fmt.Errorf("create telegram bot: %w", err)
	}
	return &Channels{bot: bot, channels: make(map[int64]*Channel)}, nil
}

func (cs *Channels) Get(id int64) messages.Channel {
	if c, ok := cs.channels[id]; ok {
		return c
	}
	c := &Channel{bot: cs.bot, chatID: id}
	cs.channels[id] = c
	return c
}

type Channel struct {
	bot    *gotgbot.Bot
	chatID int64
}

func NewChannel(token string, chatID int64) (*Channel, error) {
	bot, err := gotgbot.NewBot(token, nil)
	if err != nil {
		return nil, fmt.Errorf("create telegram bot: %w", err)
	}
	return &Channel{bot: bot, chatID: chatID}, nil
}

func (c *Channel) Send(ctx context.Context, msgs []*notes.Note) error {
	for _, msg := range msgs {
		text := msg.Content
		if text == "" {
			continue
		}
		_, err := c.bot.SendMessage(c.chatID, text, &gotgbot.SendMessageOpts{
			RequestOpts: &gotgbot.RequestOpts{
				Timeout: 10 * time.Second,
			},
		})
		if err != nil {
			return fmt.Errorf("send telegram message: %w", err)
		}
	}
	return nil
}

func (c *Channel) Recieve(ctx context.Context, offset int64) ([]*notes.Note, error) {
	updates, err := c.bot.GetUpdates(&gotgbot.GetUpdatesOpts{
		Offset:         offset,
		Timeout:        30,
		AllowedUpdates: []string{"message"},
	})
	if err != nil {
		return nil, fmt.Errorf("get telegram updates: %w", err)
	}

	var result []*notes.Note
	for _, u := range updates {
		if u.Message != nil && u.Message.Text != "" {
			result = append(result, &notes.Note{
				Content:   u.Message.Text,
				CreatedAt: time.Unix(u.Message.Date, 0),
			})
		}
		offset = u.UpdateId
	}
	return result, nil
}
