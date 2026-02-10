-- Phase 6: Spectra cloud persistence schema
-- Run this in Supabase SQL Editor or via supabase db push
--
-- Tables: experiments, runs
-- RLS enabled on all tables, enforcing owner_id = authenticated user

-- Enable UUID extension (usually already enabled on Supabase)
create extension if not exists "uuid-ossp";

-- ---------------------------------------------------------------------------
-- Experiments table
-- ---------------------------------------------------------------------------
create table if not exists experiments (
    id          uuid primary key default uuid_generate_v4(),
    owner_id    text not null,
    name        text not null,
    created_at  timestamptz not null default now(),
    config      jsonb not null default '{}'::jsonb
);

create index if not exists idx_experiments_owner on experiments(owner_id);

-- ---------------------------------------------------------------------------
-- Runs table
-- ---------------------------------------------------------------------------
create table if not exists runs (
    id              uuid primary key default uuid_generate_v4(),
    experiment_id   uuid not null references experiments(id) on delete cascade,
    owner_id        text not null,
    status          text not null check (status in ('succeeded', 'failed')),
    created_at      timestamptz not null default now(),
    results_key     text not null,
    analysis_key    text
);

create index if not exists idx_runs_experiment on runs(experiment_id);
create index if not exists idx_runs_owner on runs(owner_id);

-- ---------------------------------------------------------------------------
-- Row Level Security
-- ---------------------------------------------------------------------------

-- Enable RLS on experiments
alter table experiments enable row level security;

create policy "Users can view own experiments"
    on experiments for select
    using (owner_id = auth.uid()::text);

create policy "Users can insert own experiments"
    on experiments for insert
    with check (owner_id = auth.uid()::text);

create policy "Users can update own experiments"
    on experiments for update
    using (owner_id = auth.uid()::text);

create policy "Users can delete own experiments"
    on experiments for delete
    using (owner_id = auth.uid()::text);

-- Enable RLS on runs
alter table runs enable row level security;

create policy "Users can view own runs"
    on runs for select
    using (owner_id = auth.uid()::text);

create policy "Users can insert own runs"
    on runs for insert
    with check (owner_id = auth.uid()::text);

create policy "Users can update own runs"
    on runs for update
    using (owner_id = auth.uid()::text);

-- ---------------------------------------------------------------------------
-- Storage bucket for artifacts
-- ---------------------------------------------------------------------------
-- Create the artifacts bucket (private by default)
insert into storage.buckets (id, name, public)
values ('artifacts', 'artifacts', false)
on conflict (id) do nothing;

-- Storage RLS: users can only access their own artifact folder
create policy "Users can upload own artifacts"
    on storage.objects for insert
    with check (
        bucket_id = 'artifacts' and
        (storage.foldername(name))[1] = 'artifacts' and
        (storage.foldername(name))[2] = auth.uid()::text
    );

create policy "Users can read own artifacts"
    on storage.objects for select
    using (
        bucket_id = 'artifacts' and
        (storage.foldername(name))[1] = 'artifacts' and
        (storage.foldername(name))[2] = auth.uid()::text
    );
