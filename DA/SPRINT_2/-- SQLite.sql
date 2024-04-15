-- SQLite
SELECT COUNT(category) as count, category
FROM categories
GROUP BY category
ORDER BY count DESC

SELECT COUNT(category) as count, category
FROM categories
GROUP BY 1
ORDER BY count ASC


SELECT COUNT(category) as count, category, INSTR(category, '-')
FROM categories
GROUP BY 1





SELECT COUNT(category) as count, category, SUBSTR(category, 1, INSTR(category, '-') -1) AS cat
FROM categories
GROUP BY 1

SELECT SUBSTR(category, 1, INSTR(category, '-') -1) AS new_string 
FROM categories;



SELECT SUBSTRING(category, 1, CHARINDEX('-', category) - 1) AS new_string 
FROM categories;

SELECT * 
FROM categories
WHERE category = NULL


SELECT slug, title, podcast_id
FROM podcasts

SELECT podcast_id, title, author_id, SUBSTR(created_at, 1, INSTR(created_at, 'T') -1) AS created_on,   
FROM reviews

SELECT * 
FROM runs

SELECT COUNT(DISTINCT(author_id))
FROM reviews



