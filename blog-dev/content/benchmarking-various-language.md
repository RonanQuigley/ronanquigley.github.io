---
title: 'Language benchmarking for Web Clients & Servers'
date: 2024-05-01T08:13:34Z
draft: true
tags:
description: ''
keywords:
---

# WhatDis?

Let's take four different programming languages and write the the same web app. Candidates:

-   Python
-   Rust
-   Go
-   JavaScript (Node V8 Engine)

Some other things to mention:

-   Go/Rust are languages compiled to machine code, so are guaranteed to be faster. The question is how much faster?
-   JavaScript engine implementations these days use a JIT
-   Python is compiled to bytecode and then passed to an interpreter VM

## Methodology

There's two ways to look at this:

Virtualisation: what happens with linux in a VM that's running docker?
Native: what happens with linux running docker natively i.e no VM.

### Benchmark

The benchmarking looks like this

-   1000 RPS for all tests
-   Two containers in the same Pod using localhost to communicate. Therefore, DNS lookups are not a factor in performance
-   They are also ran with and without the the host CPU stressed out.
-   Kind is given the full compute of the host available to the one node "cluster"
-   We serve large and empty responses (more on that below)

### Data Structure Size

I've discovered from this benchmarking that the type of data structures that you're working with can have massive implications for language performance. For this, I've prepared a 3000 line JSON. For science, we can compare performance against trivial web servers/clients in the same language that do nothing but fetch/serve a 200 status code.

### Instrumentation

To measure performance, we're using prometheus. This won't be as accurate as using high procession timers in each language, but it will give us a good enough indication of the differences.
