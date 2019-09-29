DROP TABLE if exists cards.experience;

DROP TABLE if exists cards.win_rate;

DROP SCHEMA if exists cards;

CREATE SCHEMA cards AUTHORIZATION postgres;

CREATE TABLE cards.experience (
    id serial NOT NULL,
    ins_ts timestamp NOT NULL DEFAULT now(),
    agent_id varchar NULL,
    run_id varchar NOT NULL,
    vector varchar NOT NULL,
    reward numeric NOT NULL,
    action varchar NULL,
    CONSTRAINT experience_pk PRIMARY KEY (id)
);

CREATE TABLE cards.metrics (
	id bigserial NOT NULL,
	run_id varchar NOT NULL,
	win_rate numeric NOT NULL,
	win_rate_random numeric NOT NULL,
	average_reward numeric NOT NULL,
	ins_ts timestamp NOT NULL DEFAULT now(),
	CONSTRAINT win_rate_pk PRIMARY KEY (id)
);
