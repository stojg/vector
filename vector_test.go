package vector

import (
	"testing"
)

// for benchmarks always store the result to a package level variable
// so the compiler cannot eliminate the Benchmark itself.
var result *Vector3

func TestVector3_Add(t *testing.T) {
	var tests = []struct {
		a *Vector3
		b *Vector3
		c *Vector3
	}{
		{NewVector3(0, 0, 0), NewVector3(0, 0, 0), NewVector3(0, 0, 0)},
		{NewVector3(1, 2, 3), NewVector3(1, 2, 3), NewVector3(2, 4, 6)},
		{NewVector3(1, 2, 3), NewVector3(-1, -2, -3), NewVector3(0, 0, 0)},
		{NewVector3(1, 2, 3), NewVector3(-2, -3, -4), NewVector3(-1, -1, -1)},
	}
	for i, test := range tests {
		actual := test.a.Add(test.b)
		if !actual.Equals(test.c) {
			t.Errorf("TestVector3_Add: expected %v, but got %v for test %d", test.c, actual, i+1)
		}
	}
}

func BenchmarkVector3_Add(b *testing.B) {
	z := NewVector3(1, 2, 3)
	x := NewVector3(-1, -2, -3)

	b.ResetTimer()
	var r *Vector3
	for i := 0; i < b.N; i++ {
		r = z.Add(x)
	}
	result = r
}

func TestVector3_Sub(t *testing.T) {
	var tests = []struct {
		a *Vector3
		b *Vector3
		c *Vector3
	}{
		{NewVector3(0, 0, 0), NewVector3(0, 0, 0), NewVector3(0, 0, 0)},
		{NewVector3(1, 2, 3), NewVector3(-1, -2, -3), NewVector3(2, 4, 6)},
		{NewVector3(1, 2, 3), NewVector3(1, 2, 3), NewVector3(0, 0, 0)},
		{NewVector3(1, 2, 3), NewVector3(2, 3, 4), NewVector3(-1, -1, -1)},
	}
	for i, test := range tests {
		actual := test.a.Sub(test.b)
		if !actual.Equals(test.c) {
			t.Errorf("TestVector3_Sub: expected %v, but got %v for test %d", test.c, actual, i+1)
		}
	}
}

func BenchmarkVector3_Sub(b *testing.B) {
	z := NewVector3(1, 2, 3)
	x := NewVector3(-1, -2, -3)

	b.ResetTimer()
	var r *Vector3
	for i := 0; i < b.N; i++ {
		r = z.Sub(x)
	}
	result = r
}
