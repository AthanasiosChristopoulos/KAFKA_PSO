Postgres SQL:

CLI 

psql			(actual program in windows, no CMD)
\l 				(see all databases)
\c products 	(go into database products)
\dt 			(display tables of database)


Contraints:

    1) CONSTRAINT "Student_pkey" PRIMARY KEY (amka)
    2) CONSTRAINT student_fkey1 FOREIGN KEY (amka)
		  REFERENCES public."Person" (amka) MATCH SIMPLE
          ON UPDATE RESTRICT
          ON DELETE RESTRICT / ON DELETE NO ACTION / ON DELETE CASCASE

    3) CONSTRAINT "Student_am_key" UNIQUE (am)


SQL code:

1) Make it Unique:
	SELECT DISTINCT l.lab_code
    FROM "Lab" l
    ORDER BY random()
    LIMIT N;
	
3) Order Based on row index:
	SELECT * FROM "Person"
	ORDER BY ctid DESC
	LIMIT 100;

4) enums, regtype (system type) in SQL: 
    SELECT enumlabel::public.level_type
    FROM pg_enum
    WHERE enumtypid = 'public.level_type'::regtype
    ORDER BY random()
    LIMIT 1;

4) Generate Random:
	random(): Generates a random decimal number between 0.0 (inclusive) and 1.0 (exclusive) (up to 0.99999).
	SELECT floor(random() * 10 + 1)::integer; : floors that number (9.99999 => 9) and then 9 + 1 => 10
	
5) Count the number of rows:
	SELECT COUNT(*)
	FROM (	SELECT amka
			FROM "Person");
			
6)	COUNT(r.final_grade) counts => non-null, not unique instances 
	COUNT(DISTINCT r.final_grade) counts => non-null, unique instances 

7) NOT or != ...		 => WHERE s.academic_year <> year AND c.typical_season <> typical_season_1

8) Finding percentages:
	ROUND((COUNT(CASE WHEN r.final_grade >= 8.5 THEN 1 END) * 1.0 / COUNT(r.final_grade)) * 100, 2)
	-- note rounding only works with numeric
	
9) Recursive:

	a)	With Recursive
			Anc(a,d)  as ( 	
				select parent as a, child as d from ParentOf
				union
				select Anc.a, ParentOf.child as d
				from Anc, ParentOf
				where Anc.d = ParentOf.parent )
		
	b)	WITH RECURSIVE
			Anc(main, dependent) AS (
				SELECT main, dependent
				FROM "Course_depends"
				UNION
				SELECT Anc.main, cd.dependent
				FROM Anc JOIN "Course_depends" cd
				ON Anc.dependent = cd.main
			)
		SELECT main, dependent
		FROM Anc
		WHERE main = 'ΠΛΗ 101';	

9.5) UPDATE:
	UPDATE "Register" r   -- Table to Update
	SET exam_grade = 4, lab_grade = 4
	FROM "Student" st     -- Joining Table
	WHERE st.amka = '28099105058' AND r.course_code = 'ΑΓΓ 101' AND st.amka = r.amka;
		
	SELECT * 
	FROM "Student" st
	JOIN "Register" r USING(amka)
	WHERE st.amka = '28099105058' and r.course_code = 'ΑΓΓ 101';
	
9.7) INSERT:
    INSERT INTO "Person" (amka, name, father_name, surname, email)
    SELECT amka, name, father_name, surname, email FROM generated_data;
	OR
	INSERT INTO "Register" (amka, serial_number, course_code, exam_grade, lab_grade)
	VALUES ('28099105058', 1, 'ΑΓΓ 101', 6, 7);
	------------------------------------------------------------------------------------------

	The RETURNING clause is usually used with the INSERT statement to get the values of the 
	inserted rows, but it must be handled outside the CTE if you're not using temporary tables.

9.8) DELETE:
	Delete all instances:
		TRUNCATE TABLE "Person";
	------------------------------------------------------------------------------------------
	DELETE FROM "Register"
	WHERE amka = '28099105058' AND course_code = 'ΠΛΗ 102' AND serial_number = 1;
	------------------------------------------------------------------------------------------
	DELETE FROM "Register"
	WHERE ctid IN (
		SELECT ctid FROM "Register" r
		WHERE r.register_status = 'approved' and r.amka = '01010206100'
		ORDER BY ctid DESC
		LIMIT 2
	);
	------------------------------------------------------------------------------------------
	ALTER TABLE "LabTeacher" DROP CONSTRAINT labStaff_person_fkey;
	ALTER TABLE "LabTeacher"
	ADD CONSTRAINT labStaff_person_fkey
	FOREIGN KEY (amka) REFERENCES "Person"(amka)
	ON DELETE CASCADE;

9.82) DROP:
	DROP FUNCTION IF EXISTS find_max_professor(integer,semester_season_type);
	CREATE OR REPLACE FUNCTION find_max_professor(year integer, typical_season_1 semester_season_type)

9.9) ALTER:
	ALTER TABLE "Course"
	ALTER COLUMN course_code TYPE CHARACTER VARYING;
	------------------------------------------------------------------------------------------
	ALTER TABLE "Professor" DROP CONSTRAINT "Professor_labJoins_fkey";
	ALTER TABLE "Professor"
	ADD CONSTRAINT "Professor_labJoins_fkey"
	FOREIGN KEY (labjoins) REFERENCES "Lab"(lab_code)
		MATCH SIMPLE
		ON UPDATE NO ACTION
		ON DELETE CASCADE;


10) Triggers:

CREATE OR REPLACE FUNCTION student_update() RETURNS trigger AS
$BODY$
BEGIN
	IF (TG_OP = 'DELETE') THEN
		   INSERT INTO my_student_audit(operation,operation_time,am,message) 
		   SELECT 'D', now(), OLD.am, CONCAT('Deletion Attempted: ', OLD.name,' ',OLD.surname);
		   RETURN NULL;
	ELSIF (TG_OP = 'UPDATE') THEN
		   IF (NEW.am <> OLD.am) THEN
			INSERT INTO my_student_audit(operation,operation_time,am,message) 
			SELECT 'U', now(), NEW.am, CONCAT('AM Update denied:',' ',NEW.am);
			RETURN NULL;
		   ELSE
			INSERT INTO my_student_audit(operation,operation_time,am,message) 
			SELECT 'U', now(), NEW.am, CONCAT(NEW.name,' ',NEW.surname);
			NEW.surname = OLD.surname;
			RETURN NEW;
		   END IF;
	ELSIF (TG_OP = 'INSERT') THEN
		   IF (CAST( date_part('year',now()) as text ) = left(NEW.am,4)) THEN
			insert into my_student_audit(operation,operation_time,am,message) 
			SELECT 'I', now(), NEW.am, CONCAT(NEW.name,' ',NEW.surname);
			RETURN NEW;
		   ELSE
			RAISE EXCEPTION 'Invalid AM';
		   END IF;
	END IF;
END;
$BODY$
LANGUAGE plpgsql;

CREATE TRIGGER student_monitor BEFORE INSERT OR UPDATE OR DELETE ON "myStudent" 
FOR EACH ROW EXECUTE PROCEDURE student_update(); 

11) Filter Using Regex:
	SELECT * 
	FROM "Course" c
	WHERE c.course_code LIKE '%ΑΓΓ 101%';
	
12) Debugging:
	RAISE NOTICE 'AAAAA';
	
13) Fast Functions: (allows for PL/pgSQL blocks execution)

DO $$ 
DECLARE
    num_of_required_courses integer;
    number_of_passed_courses integer;
BEGIN
    -- Count all required courses
    SELECT COUNT(c.course_code) INTO num_of_required_courses
    FROM "Course" c
    WHERE c.obligatory = True;

    -- Count all passed required courses for the student
    SELECT COUNT(r.course_code) INTO number_of_passed_courses
    FROM "Course" c
    JOIN "Register" r ON c.course_code = r.course_code  
    WHERE r.amka = '01010206100' AND c.obligatory = True AND r.register_status = 'pass';

    RAISE NOTICE 'num_of_required_courses: %', num_of_required_courses;
    RAISE NOTICE 'number_of_passed_courses: %', number_of_passed_courses;
END $$ LANGUAGE plpgsql;

14) VOLATILE => i dont expect deterministic output

15) Sequences:
		CREATE SEQUENCE IF NOT EXISTS amka_sequence;
		ALTER SEQUENCE amka_sequence RESTART WITH 10000000000;
		nextval('amka_sequence')
		
		CREATE SEQUENCE IF NOT EXISTS diploma_id_sequence;
		ALTER SEQUENCE diploma_id_sequence RESTART WITH 10000000000;
		nextval('diploma_id_sequence')
		
16) COALESCE:
	COALESCE(SUM(c.lecture_hours), 0), replace any potential NULL values with 0
	
17) WITH:
	-- SQL Statement
	select * from 
	(select amka, count(*) as c from "Register"
		where course_code =  'ΠΛΗ 101'
		group by amka) as t1
	join 	--or natural join
	(select max(c)as c from
		(select count(*) as c from "Register"
		where course_code =  'ΠΛΗ 101'
		group by amka) as t2) as t3 using (c)
	natural join "Student"

	-- SQL Statement with CTE
		
	WITH
	regp101 as 
		(select amka, count(*) as c from "Register"
		where course_code =  'ΠΛΗ 101'
		group by amka),
	maxregp as (select max(c) as c from regp101)	

	select * from
	regp101 natural join maxregp natural join "Student"
	 
18) Sequences:


-- CREATE SEQUENCE IF NOT EXISTS amka_sequence;
-- ALTER SEQUENCE amka_sequence RESTART WITH 10000000000;
-- -- nextval('amka_sequence')		
		    
-----------------------------------------------------------------------------
Optimization:

1) CREATE INDEX Person_surname_idx ON "Person" USING btree(surname); (btree, hash, ... : are indexing methods)

2) 
	EXPLAIN SELECT * FROM "Person" ...
	vs
	EXPLAIN ANALYSE SELECT * FROM "Person" ...

3) CLUSTER "Person" USING Person_name_idx; 

4) Διαφοροι αλγοριθμοι join (συνδεσεων):
	
	set enable_mergejoin=off; -- on
	set enable_hashjoin=off;  -- on

5) SET max_parallel_workers_per_gather = 0;