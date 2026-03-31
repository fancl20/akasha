package notes

import (
	"iter"
	"time"
)

type NoteID int64

type Note struct {
	ID          NoteID
	Content     string
	CreatedAt   time.Time
	DerivedFrom Notes
}

type Notes interface {
	Iter() iter.Seq[*Note]
	Err() error
}

type Ref interface {
	Append(n *Note) error
	Notes() Notes
}

type Refs interface {
	Get(ref string) (Ref, error)
	List() ([]string, error)
}
