package messages

import (
	"context"

	"github.com/fancl20/akasha/pkg/notes"
)

type Channel interface {
	Send(ctx context.Context, msgs []*notes.Note) error
	Recieve(ctx context.Context, offset int64) ([]*notes.Note, error)
}

type Channels interface {
	Get(id int64) Channel
}
