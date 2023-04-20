use std::fs::File;
use std::io::{BufReader, BufRead, Write};
use std::io;
use std::time::Duration;
use std::thread;

use itertools::Itertools;

fn main() {
    // This delay is sadly necessary as Windows apparently can't deal with the prints to console otherwise
    thread::sleep(Duration::new(0, 100000000));

    for i in 0..=9 {
        let mut stack = file_to_stack(format!("inputs/pancake{}.txt", i).as_str());
        println!("\nBeispiel {}:", i + 1);
        println!("Stapel: {:?}", stack.pancakes);
        println!("PWUE: {}, Höhe {}", Stack::pwue(stack.height), stack.height);
        stack.sort();
        stack.print();
        match stack.to_file(format!("inputs/pancake{}_res.txt", i).as_str()) {
            Err(_) => println!("WARNING: Writing to file went wrong, please make sure the leading path exists"),
            _ => ()
        }
    }

    loop {
        println!("\nGib nun ein eigenes Beispiel an (z.B.: 2, 3, 1) oder einzelne Zahlen für den Worst-Stack:");
        let mut raw_input = String::new();
        io::stdin().read_line(&mut raw_input).expect("error: unable to read user input");
        let input = raw_input[0..raw_input.len() - 2].split(", ").map(|a| a.to_string()).collect_vec();

        if input.len() != 1 {
            let mut stack = Stack::new(In::S(input));
            println!("PWUE: {}, Höhe {}", Stack::pwue(stack.height), stack.height);
            stack.sort();
            stack.print();
        } else {
            let height = input[0].parse::<usize>().unwrap();
            println!("PWUE: {}", Stack::pwue(height));
            println!("Worst-Stack: {:?}", Stack::worst_stack(height).pancakes);
        }
    }
}

#[derive(Debug)]
struct Stack {
    height: usize,
    pancakes: Vec<usize>,
    flips: (usize, Vec<usize>)
}

impl Stack {
    pub fn print(&self) {
        println!("WUEOs: {:?} bei {:?}\nErgebnisstapel: {:?}", self.flips.0, self.flips.1, self.pancakes);
    }

    pub fn data(&self) -> String {
        format!("WUEOs: {:?} bei {:?}\nErgebnisstapel: {:?}", self.flips.0, self.flips.1, self.pancakes)
    } 

    pub fn new(items: In) -> Stack {
        match items {
            In::S(items) => Stack {height: items.len(), pancakes: items.into_iter().map(|a| a.parse::<usize>().unwrap()).collect(), flips: (0, Vec::new())},
            In::I(items) => Stack {height: items.len(), pancakes: items.into_iter().collect(), flips: (0, Vec::new())}
        }
    }

    pub fn len(&self) -> usize {self.pancakes.len()}

    pub fn sort(&mut self) {
        while !self.is_sorted() {
            if self.pancakes.get(0).unwrap() == &self.biggest_unsorted().0 {
                self.flip(self.len() - self.amount_sorted());
            } else {
                self.flip(self.biggest_unsorted().1 + 2);
            }
        }
    }

    // flips UNDER the given pos and "eats" the top pancake after
    pub fn flip(&mut self, pos: usize) {
        self.flips.0 += 1;
        self.flips.1.push(pos);
        let stack = self.pancakes.clone().into_iter();
        let mut flipped = stack.clone().take(pos - 1).rev().collect::<Vec<usize>>();
        let mut rest = stack.skip(pos).collect::<Vec<usize>>();
        flipped.append(&mut rest);
        self.pancakes = flipped;
    }

    pub fn is_sorted(&self) -> bool {
        let mut sorted = self.pancakes.clone();
        sorted.sort();
        return self.pancakes == sorted
    }

    pub fn amount_sorted(&self) -> usize {
        let stack = self.pancakes.clone();
        let mut sorted = self.pancakes.clone();
        sorted.sort();
        
        let mut sorted_iter = sorted.iter().rev();
        let stack_iter = stack.iter().rev();

        stack_iter.take_while(|x| *x == sorted_iter.next().unwrap()).collect::<Vec<&usize>>().len()
    }

    pub fn biggest_unsorted(&self) -> (usize, usize) {
        let stack = self.pancakes.clone();
        
        let biggest_unsorted = *stack.iter().take(stack.len() - self.amount_sorted() - 1).max().unwrap();
        let pos_biggest = stack.iter().position(|a| *a == biggest_unsorted).unwrap(); 

        (biggest_unsorted, pos_biggest)
    }

    pub fn worst_stack(size: usize) -> Stack {
        let mut items = Vec::new();
        let mid = size / 2;
        for i in 1.. {
            if i != 0 {items.push(i)}
            if items.len() == size {break;}
            if size - i + 1 > mid {items.push(size - i + 1)}
            if items.len() == size {break;}
        }
        items = items.into_iter().rev().collect();
        Stack::new(In::I(items))
    }

    pub fn pwue(size: usize) -> usize {
        return ((size + 1) as f32 / 2.) as usize
    }

    pub fn to_file(&self, filepath: &str) -> Result<(), std::io::Error> {
        let mut file = File::create(filepath)?;
        file.write_all(&self.data().as_bytes())?;
        Ok(())
    }

}

fn file_to_stack(filename: &str) -> Stack {
    let file = File::open(filename)
    .expect(format!("can't open inputfile").as_str());

    let mut lines = BufReader::new(file).lines()
        .filter_map(|l| l.ok());

    lines.next();

    Stack::new(In::S(lines.collect()))
}

// So i can conveniently make new stacks from both usizes and Strings
#[derive(Debug)]
enum In {
    S(Vec<String>),
    I(Vec<usize>)
}