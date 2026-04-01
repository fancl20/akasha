package notes

import (
	"crypto/sha256"
	"database/sql"
	_ "embed"
	"encoding/hex"
	"errors"
	"fmt"
	"iter"
	"time"

	_ "modernc.org/sqlite"
)

//go:embed db.sql
var schema string

type Storage struct {
	db *sql.DB
}

func Open(dsn string) (*Storage, error) {
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open database: %w", err)
	}
	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("create schema: %w", err)
	}
	return &Storage{db: db}, nil
}

func (s *Storage) Refs() Refs { return &refs{s: s} }

func contentHash(body string) string {
	h := sha256.Sum256([]byte(body))
	return hex.EncodeToString(h[:])
}

func (s *Storage) ensureContent(tx *sql.Tx, body string) (string, error) {
	hash := contentHash(body)
	if _, err := tx.Exec("INSERT OR IGNORE INTO contents (hash, body) VALUES (?, ?)", hash, body); err != nil {
		return "", fmt.Errorf("insert content: %w", err)
	}
	return hash, nil
}

func (s *Storage) loadNote(id NoteID) (*Note, NoteID, error) {
	n := &Note{ID: id}
	var createdAt, prev, derivedTail, derivedHead sql.NullInt64
	if err := s.db.QueryRow(
		"SELECT c.body, n.created_at, n.prev, n.derived_tail, n.derived_head FROM notes n JOIN contents c ON n.content = c.hash WHERE n.id = ?",
		id,
	).Scan(&n.Content, &createdAt, &prev, &derivedTail, &derivedHead); err != nil {
		return nil, 0, fmt.Errorf("load note %d: %w", id, err)
	}

	n.CreatedAt = time.Unix(createdAt.Int64, 0)
	if derivedTail.Valid {
		n.DerivedFrom = &notes{s: s, tailID: NoteID(derivedTail.Int64), headID: NoteID(derivedHead.Int64)}
	}
	return n, NoteID(prev.Int64), nil
}

type notes struct {
	s      *Storage
	tailID NoteID
	headID NoteID
	err    error
}

func (ns *notes) Iter() iter.Seq[*Note] {
	return func(yield func(*Note) bool) {

		for cur := ns.tailID; cur != ns.headID; {
			n, prev, err := ns.s.loadNote(cur)
			if err != nil {
				ns.err = err
				return
			}
			if !yield(n) {
				return
			}
			cur = prev
		}
	}
}

func (ns *notes) Err() error { return ns.err }

type ref struct {
	s      *Storage
	name   string
	tailID NoteID
}

func (r *ref) Append(n *Note) error {
	body := n.Content
	tx, err := r.s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback()

	hash, err := r.s.ensureContent(tx, body)
	if err != nil {
		return err
	}

	var prev any
	if r.tailID != 0 {
		prev = r.tailID
	}

	var derivedTail, derivedHead any
	if n.DerivedFrom != nil {
		dn := n.DerivedFrom.(*notes)
		derivedTail = dn.tailID
		derivedHead = dn.headID
	}
	result, err := tx.Exec(
		"INSERT INTO notes (content, created_at, prev, derived_tail, derived_head) VALUES (?, ?, ?, ?, ?)",
		hash, n.CreatedAt.Unix(), prev, derivedTail, derivedHead,
	)
	if err != nil {
		return fmt.Errorf("insert note: %w", err)
	}
	id, err := result.LastInsertId()
	if err != nil {
		return fmt.Errorf("last insert id: %w", err)
	}

	if r.name != "" {
		if _, err = tx.Exec(
			"INSERT OR REPLACE INTO refs (ref, tail) VALUES (?, ?)",
			r.name, id,
		); err != nil {
			return fmt.Errorf("update ref: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit: %w", err)
	}
	r.tailID = NoteID(id)
	return nil
}

func (r *ref) Notes() Notes {
	if r.tailID == 0 {
		return &notes{s: r.s}
	}
	return &notes{s: r.s, tailID: r.tailID}
}

type refs struct {
	s *Storage
}

func (rs *refs) Get(name string) (Ref, error) {
	if name == "" {
		return &ref{s: rs.s}, nil
	}

	var tail int64
	if err := rs.s.db.QueryRow("SELECT tail FROM refs WHERE ref = ?", name).Scan(&tail); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return &ref{s: rs.s, name: name}, nil
		}
		return nil, err
	}
	return &ref{s: rs.s, name: name, tailID: NoteID(tail)}, nil
}

func (rs *refs) List() ([]string, error) {
	rows, err := rs.s.db.Query("SELECT ref FROM refs ORDER BY ref")
	if err != nil {
		return nil, fmt.Errorf("list refs: %w", err)
	}
	defer rows.Close()

	var result []string
	for rows.Next() {
		var r string
		if err := rows.Scan(&r); err != nil {
			return nil, fmt.Errorf("scan ref: %w", err)
		}
		result = append(result, r)
	}
	return result, rows.Err()
}
