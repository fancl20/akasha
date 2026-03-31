package notes

import (
	"slices"
	"testing"
)

func openTestDB(t *testing.T) *Storage {
	t.Helper()
	s, err := Open(":memory:")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { s.db.Close() })
	return s
}

// newNote creates a *Note with the given content for testing.
func newNote(body string) *Note {
	return &Note{Content: body}
}

func TestRefsGetNew(t *testing.T) {
	s := openTestDB(t)
	r, err := s.Refs().Get("test-ref")
	if err != nil {
		t.Fatalf("Get new ref: %v", err)
	}
	if r == nil {
		t.Fatal("Get new ref: got nil")
	}
	// New ref should have no notes
	notes := r.Notes()
	if notes == nil {
		t.Fatal("Notes() returned nil for new ref")
	}
	var collected []*Note
	for n := range notes.Iter() {
		collected = append(collected, n)
	}
	if err := notes.Err(); err != nil {
		t.Fatalf("Notes.Err(): %v", err)
	}
	if len(collected) != 0 {
		t.Fatalf("new ref should have no notes, got %d", len(collected))
	}
}

func TestRefsGetExisting(t *testing.T) {
	s := openTestDB(t)
	rs := s.Refs()

	r, err := rs.Get("myref")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}

	// Append a note so the ref is persisted
	body := "hello world"
	if err := r.Append(newNote(body)); err != nil {
		t.Fatalf("Append: %v", err)
	}

	// Get the same ref again
	r2, err := rs.Get("myref")
	if err != nil {
		t.Fatalf("Get existing: %v", err)
	}

	notes := r2.Notes()
	var collected []*Note
	for n := range notes.Iter() {
		collected = append(collected, n)
	}
	if len(collected) != 1 {
		t.Fatalf("existing ref should have 1 note, got %d", len(collected))
	}
	if collected[0].Content != body {
		t.Fatalf("note content = %q, want %q", collected[0].Content, body)
	}
}

func TestAppendMultipleNotes(t *testing.T) {
	s := openTestDB(t)
	r, err := s.Refs().Get("chain")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}

	bodies := []string{"first", "second", "third"}
	for _, b := range bodies {
		if err := r.Append(newNote(b)); err != nil {
			t.Fatalf("Append %q: %v", b, err)
		}
	}

	notes := r.Notes()
	var collected []*Note
	for n := range notes.Iter() {
		collected = append(collected, n)
	}
	if notes.Err() != nil {
		t.Fatalf("Notes.Err(): %v", notes.Err())
	}

	// Notes are linked tail→prev, so iteration yields newest first
	if len(collected) != 3 {
		t.Fatalf("got %d notes, want 3", len(collected))
	}
	got := make([]string, len(collected))
	for i, n := range collected {
		got[i] = n.Content
	}
	want := []string{"third", "second", "first"}
	if !slices.Equal(got, want) {
		t.Fatalf("notes order = %v, want %v", got, want)
	}

	// Verify IDs are assigned and increasing
	for i := 1; i < len(collected); i++ {
		if collected[i].ID >= collected[i-1].ID {
			t.Fatalf("note IDs not decreasing: notes[%d].ID=%d >= notes[%d].ID=%d",
				i, collected[i].ID, i-1, collected[i-1].ID)
		}
	}
}

func TestRefList(t *testing.T) {
	s := openTestDB(t)
	rs := s.Refs()

	// Empty list
	list, err := rs.List()
	if err != nil {
		t.Fatalf("List empty: %v", err)
	}
	if len(list) != 0 {
		t.Fatalf("empty List = %v, want []", list)
	}

	// Create multiple refs by appending a note to each
	for _, name := range []string{"beta", "alpha", "gamma"} {
		r, err := rs.Get(name)
		if err != nil {
			t.Fatalf("Get %q: %v", name, err)
		}
		if err := r.Append(newNote("note for " + name)); err != nil {
			t.Fatalf("Append to %q: %v", name, err)
		}
	}

	list, err = rs.List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	want := []string{"alpha", "beta", "gamma"}
	if !slices.Equal(list, want) {
		t.Fatalf("List = %v, want %v", list, want)
	}
}

func TestDerivedFrom(t *testing.T) {
	s := openTestDB(t)
	rs := s.Refs()

	// Create a source ref with notes
	src, err := rs.Get("source")
	if err != nil {
		t.Fatalf("Get source: %v", err)
	}
	if err := src.Append(newNote("src-a")); err != nil {
		t.Fatalf("Append src-a: %v", err)
	}
	if err := src.Append(newNote("src-b")); err != nil {
		t.Fatalf("Append src-b: %v", err)
	}

	// Append a derived note referencing source notes
	dst, err := rs.Get("dest")
	if err != nil {
		t.Fatalf("Get dest: %v", err)
	}
	derived := &Note{Content: "derived", DerivedFrom: src.Notes()}
	if err := dst.Append(derived); err != nil {
		t.Fatalf("Append derived: %v", err)
	}

	// Reload dest and verify DerivedFrom was persisted
	r2, err := rs.Get("dest")
	if err != nil {
		t.Fatalf("Get dest again: %v", err)
	}
	ns := r2.Notes()
	var derivedNote *Note
	for n := range ns.Iter() {
		derivedNote = n
		break
	}
	if derivedNote == nil {
		t.Fatal("no note found in dest")
	}
	if derivedNote.Content != "derived" {
		t.Fatalf("content = %q, want %q", derivedNote.Content, "derived")
	}
	if derivedNote.DerivedFrom == nil {
		t.Fatal("DerivedFrom is nil, should reference source notes")
	}

	// Verify we can iterate the derived notes
	var derivedContents []string
	for n := range derivedNote.DerivedFrom.Iter() {
		derivedContents = append(derivedContents, n.Content)
	}
	if err := derivedNote.DerivedFrom.Err(); err != nil {
		t.Fatalf("DerivedFrom.Err(): %v", err)
	}
	want := []string{"src-b", "src-a"}
	if !slices.Equal(derivedContents, want) {
		t.Fatalf("derived notes = %v, want %v", derivedContents, want)
	}
}

func TestAnonymousRef(t *testing.T) {
	s := openTestDB(t)
	rs := s.Refs()

	r, err := rs.Get("")
	if err != nil {
		t.Fatalf("Get anonymous: %v", err)
	}
	if r == nil {
		t.Fatal("Get anonymous: got nil")
	}

	// Append notes to anonymous ref
	bodies := []string{"anon-first", "anon-second"}
	for _, b := range bodies {
		if err := r.Append(newNote(b)); err != nil {
			t.Fatalf("Append %q: %v", b, err)
		}
	}

	// Notes should be readable from the ref
	notes := r.Notes()
	var collected []*Note
	for n := range notes.Iter() {
		collected = append(collected, n)
	}
	if notes.Err() != nil {
		t.Fatalf("Notes.Err(): %v", notes.Err())
	}
	if len(collected) != 2 {
		t.Fatalf("got %d notes, want 2", len(collected))
	}
	got := make([]string, len(collected))
	for i, n := range collected {
		got[i] = n.Content
	}
	want := []string{"anon-second", "anon-first"}
	if !slices.Equal(got, want) {
		t.Fatalf("notes order = %v, want %v", got, want)
	}

	// Anonymous ref should not appear in List
	list, err := rs.List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(list) != 0 {
		t.Fatalf("List should be empty for anonymous refs, got %v", list)
	}
}
